/* Copyright 2023 COS Lab at Kyung Hee University. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(CORE_CM7)

#include <TensorFlowLite.h>

#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

#include "tinyseg_transfer_module.h"

namespace tflite {

namespace {

const int kMaxInputNum = 10;
const int kOutputTensor = 0;

struct TensorInfo {
  uint8_t id;
  bool fetching;
};

struct OpData {
  uint8_t axis;
  uint8_t victim;
  int num_inputs;
  TensorInfo* inputs;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* output_tensor =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output_tensor != nullptr);
  TfLiteType output_type = output_tensor->type;

  // Retrieve the information for fetching tensors. 
  TFLITE_DCHECK(node->custom_initial_data != nullptr); 
  uint8_t* custom_initial_data = (uint8_t*) node->custom_initial_data;

  const int num_inputs = NumInputs(node);
  int num_fetches = (node->custom_initial_data_size - 1) / 2;
  int actual_num_inputs = num_inputs + num_fetches;

  TF_LITE_ENSURE(context, actual_num_inputs <= kMaxInputNum);

  OpData* data = static_cast<OpData*>(node->user_data);
  data->axis = custom_initial_data[0];
  data->num_inputs = actual_num_inputs;

  // Initialize tensor information
  data->inputs = reinterpret_cast<TensorInfo*>(context->AllocatePersistentBuffer(
     context, actual_num_inputs * sizeof(TensorInfo*)));
  for (int i = 0; i < actual_num_inputs; i++) {
    data->inputs[i].id = 0;
    data->inputs[i].fetching = false;
  }

  for (int i = 0; i < num_fetches; ++i) {
    uint8_t tensor_pos = custom_initial_data[i*2+1];
    uint8_t tensor_id = custom_initial_data[i*2+2];

    data->inputs[tensor_pos].id = tensor_id;
    data->inputs[tensor_pos].fetching = true;
 
    TfLiteTensor* input_tensor = micro_context->AllocateTempTfLiteTensor(tensor_id);
    TfLiteType input_type = input_tensor->type;
    TF_LITE_ENSURE_EQ(context, output_type, input_type);

    micro_context->DeallocateTempTfLiteTensor(input_tensor);
  }
 
  int victim_size = 0;
  int victim = -1;

  int count = 0; 
  for (int i = 0; i < actual_num_inputs; i++) {
    if (!data->inputs[i].fetching) {
      data->inputs[i].id = node->inputs->data[count++];
 
      TfLiteTensor* input_tensor = micro_context->AllocateTempTfLiteTensor(data->inputs[i].id);
      TfLiteType input_type = input_tensor->type;
      TF_LITE_ENSURE_EQ(context, output_type, input_type);
 
      int elem_count = ElementCount(*input_tensor->dims);
      if (elem_count > victim_size) {
        victim = i;
        victim_size = elem_count;
      }
 
      micro_context->DeallocateTempTfLiteTensor(input_tensor);
    }
  }

  data->victim = static_cast<int8_t>(victim);

  switch (output_type) {
    case kTfLiteFloat32:
    case kTfLiteInt16:
    case kTfLiteInt8:
      break;
    default:
      MicroPrintf("Op Fetching does not currently support Type '%s'.",
                  TfLiteTypeGetName(output_type));
      return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(output_tensor);
  return kTfLiteOk;
}

template <typename Scalar>
inline void Fetching(const OpData* data,
                     const RuntimeShape* inputs_shape,
                     Scalar* const* inputs_data,
                     const RuntimeShape& output_shape,
                     Scalar* output_data) {
  uint8_t axis = data->axis;
  uint8_t num_inputs = data->num_inputs;
  const int concat_dimensions = output_shape.DimensionsCount();
  TFLITE_DCHECK_LT(axis, concat_dimensions);

  int64_t concat_size = 0;
  for (int i = 0; i < num_inputs; i++) {
    TFLITE_DCHECK_EQ(inputs_shape[i].DimensionsCount(), concat_dimensions);
    for (int j = 0; j < concat_dimensions; j++) {
      if (j != axis) {
        MatchingDim(inputs_shape[i], j, output_shape, j);
      }
    }
    concat_size += inputs_shape[i].Dims(axis);
  }
  TFLITE_DCHECK_EQ(concat_size, output_shape.Dims(axis));

  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= output_shape.Dims(i);
  }
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < concat_dimensions; ++i) {
    base_inner_size *= output_shape.Dims(i);
  }

  Scalar* output_ptr = output_data;
  for (int k = 0; k < outer_size; ++k) {
    for (int i = 0; i < num_inputs; ++i) {
      const int copy_size = inputs_shape[i].Dims(axis) * base_inner_size;
      if (!data->inputs[i].fetching) {
        const Scalar* input_ptr = inputs_data[i] + k * copy_size;
        memcpy(output_ptr, input_ptr, copy_size * sizeof(Scalar));
      }
      output_ptr += copy_size;
    }
  }

  int output_offset = 0;

  const int victim_size = inputs_shape[data->victim].FlatSize();
  const int step_size = output_shape.Dims(axis) * base_inner_size;

  for (int i = 0; i < num_inputs; ++i) {
    const int copy_size = inputs_shape[i].Dims(axis) * base_inner_size;
    if (data->inputs[i].fetching) {
      Scalar* output_ptr = output_data + output_offset;
      Scalar* victim_ptr = inputs_data[data->victim];
      int victim_offset = 0;

      const int input_size = inputs_shape[i].FlatSize();
      for (int k = 0; k < input_size; k++) {
        if (k % victim_size == 0) {
          int transfer_size = std::min(victim_size, input_size - victim_size) * sizeof(Scalar);
          transfer_module->transfer_in(data->inputs[i].id, (uint8_t*) victim_ptr, transfer_size);
          victim_offset = 0;
        }
        output_ptr[k % copy_size] = victim_ptr[victim_offset++];
        if (k % copy_size == copy_size - 1) {
          output_ptr += step_size;
        }
      }
    }
    output_offset += copy_size;
  }

  return;
}

inline void GetAllInputTensorShapes(const TfLiteContext* context,
                                    const TensorInfo* tensors,
                                    const int num_tensors,
                                    RuntimeShape shapes[kMaxInputNum]) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(tensors != nullptr);
  for (int i = 0; i < num_tensors; ++i) {
    const TfLiteEvalTensor* t = context->GetEvalTensor(context, tensors[i].id);
    RuntimeShape shape = tflite::micro::GetTensorShape(t);
    shapes[i].ReplaceWith(shape.DimensionsCount(), shape.DimsData());
  }
}

template <typename T>
inline void GetAllInputTensorData(const TfLiteContext* context,
                                  const TensorInfo* tensors,
                                  const int num_tensors,
                                  T* data[kMaxInputNum]) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(tensors != nullptr);
  for (int i = 0; i < num_tensors; ++i) {
    if (!tensors[i].fetching) {
      TfLiteEvalTensor* t = context->GetEvalTensor(context, tensors[i].id);
      data[i] = tflite::micro::GetTensorData<T>(t);
    }
  }
}

template <typename data_type>
void EvalImpl(TfLiteContext* context, TfLiteNode* node) {
  RuntimeShape inputs_shape[kMaxInputNum];
  data_type* inputs_data[kMaxInputNum];

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  GetAllInputTensorShapes(context, data->inputs, data->num_inputs, inputs_shape);
  GetAllInputTensorData(context, data->inputs, data->num_inputs, inputs_data);

  TfLiteEvalTensor* output =
    tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  Fetching(data, inputs_shape, inputs_data,
           tflite::micro::GetTensorShape(output),
           tflite::micro::GetTensorData<data_type>(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteEvalTensor* output_tensor =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TFLITE_DCHECK(output_tensor != nullptr);
  TfLiteType output_type = output_tensor->type;

  switch (output_type) {
    case kTfLiteFloat32:
      EvalImpl<float>(context, node);
      break;
    case kTfLiteInt16:
      EvalImpl<int16_t>(context, node);
      break;
    case kTfLiteInt8:
      EvalImpl<int8_t>(context, node);
      break;
    default:
      MicroPrintf("Op Fetching does not currently support Type '%s'.",
                  TfLiteTypeGetName(output_type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace.

TfLiteRegistration* Register_FETCHING() {
  static TfLiteRegistration r = tflite::micro::RegisterOp(Init, Prepare, Eval);
  return &r;
}

}

#endif
