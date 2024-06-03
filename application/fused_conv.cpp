/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/kernels/padding.h"

#include "third_party/cmsis_nn/Include/arm_nn_types.h"
#include "third_party/cmsis_nn/Include/arm_nn_math_types.h"
#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

#include "tinyseg_transfer_module.h"

arm_cmsis_nn_status arm_fused_convolve_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params *conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims inputs_dims[],
                                          const q7_t *inputs_data[],
                                          int32_t inputs_offset[],
                                          int num_inputs,
                                          const cmsis_nn_dims *filter_dims,
                                          const q7_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          q7_t *output_data,
                                          q7_t *buffer_tmp);

namespace tflite {
namespace {

const int kMaxInputNum = 10;
const int kConvOutputTensor = 0;

struct TensorInfo {
  uint8_t id;
  bool fetching;
  int32_t zero_point;
};

struct OpData {
  TfLitePaddingValues padding;
  int stride_width;
  int stride_height;
  int activation;
  int dilation_width_factor;
  int dilation_height_factor;
  bool use_bias;
  int num_inputs;
  TensorInfo* inputs;
  int buffer_idx;
  int scratch_idx;
  int32_t output_zero_point;
  int32_t *output_multiplier;
  int32_t *output_shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

bool CheckIfUseBias(MicroContext* context, TfLiteNode* node, int num_inputs) {
  TfLiteTensor* last = context->AllocateTempInputTensor(node, num_inputs - 1); 
  int dims_size = last->dims->size;
  context->DeallocateTempTfLiteTensor(last);
  return dims_size == 1;
}

TfLitePadding ConvertPadding(int padding) {
  switch (padding) {
    case Padding_SAME:
      return kTfLitePaddingSame;
    case Padding_VALID:
      return kTfLitePaddingValid;
  }
  return kTfLitePaddingUnknown;
}

TfLiteFusedActivation ConvertActivation(int activation) {
  switch (activation) {
    case ActivationFunctionType_NONE:
      return kTfLiteActNone;
    case ActivationFunctionType_RELU:
      return kTfLiteActRelu;
    case ActivationFunctionType_RELU_N1_TO_1:
      return kTfLiteActReluN1To1;
    case ActivationFunctionType_RELU6:
      return kTfLiteActRelu6;
    case ActivationFunctionType_TANH:
      return kTfLiteActTanh;
    case ActivationFunctionType_SIGN_BIT:
      return kTfLiteActSignBit;
  }
  return kTfLiteActNone;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  
  OpData* data = static_cast<OpData*>(node->user_data);
  MicroContext* micro_context = GetMicroContext(context);

  TFLITE_DCHECK(node->custom_initial_data != nullptr); 
  uint8_t* custom_initial_data = (uint8_t*) node->custom_initial_data;

  int padding_option = custom_initial_data[0];
  data->stride_width = custom_initial_data[1];
  data->stride_height = custom_initial_data[2];
  data->activation = custom_initial_data[3];
  data->dilation_width_factor = custom_initial_data[4];
  data->dilation_height_factor = custom_initial_data[5];

  uint8_t axis = custom_initial_data[6];
  TF_LITE_ENSURE_MSG(context, axis == 3, 
    "Op FusedConv2D currently supports concatenation along the channel axis only.");

  const int num_inputs = NumInputs(node);
  data->use_bias = CheckIfUseBias(micro_context, node, num_inputs);

  int num_fetches = (node->custom_initial_data_size - 7) / 2;
  TF_LITE_ENSURE_MSG(context, num_fetches == 1, 
    "Op FusedConv2D currently supports fetching a single tensor only.");

  int actual_num_inputs = num_inputs + num_fetches - (data->use_bias + 1);
  TF_LITE_ENSURE(context, actual_num_inputs <= kMaxInputNum);

  data->num_inputs = actual_num_inputs;
  data->inputs = reinterpret_cast<TensorInfo*>(context->AllocatePersistentBuffer(
     context, actual_num_inputs * sizeof(TensorInfo*)));
  for (int i = 0; i < actual_num_inputs; i++) {
    data->inputs[i].id = 0;
    data->inputs[i].fetching = false;
  }

  TfLiteTensor* inputs[kMaxInputNum];
  for (int i = 0; i < num_fetches; ++i) {
    uint8_t tensor_pos = custom_initial_data[i*2+7];
    uint8_t tensor_id = custom_initial_data[i*2+8];

    data->inputs[tensor_pos].id = tensor_id;
    data->inputs[tensor_pos].fetching = true;
 
    inputs[tensor_pos] = micro_context->AllocateTempTfLiteTensor(tensor_id);
    TF_LITE_ENSURE(context, inputs[tensor_pos] != nullptr);
    data->inputs[tensor_pos].zero_point = inputs[tensor_pos]->params.zero_point;
  }

  int count = 0;
  for (int i = 0; i < actual_num_inputs; i++) {
    if (!data->inputs[i].fetching) {
      data->inputs[i].id = node->inputs->data[count++];

      inputs[i] = micro_context->AllocateTempTfLiteTensor(data->inputs[i].id);
      TF_LITE_ENSURE(context, inputs[i] != nullptr);
      data->inputs[i].zero_point = inputs[i]->params.zero_point;
    }
  }

  const int kConvFilterTensor = num_inputs - (data->use_bias + 1);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(node, kConvFilterTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  TfLiteTensor* bias = (data->use_bias)
    ? micro_context->AllocateTempInputTensor(node, kConvFilterTensor + 1)
    : nullptr;

  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  cmsis_nn_dims input_dims;
  input_dims.n = inputs[0]->dims->data[0];
  input_dims.h = inputs[0]->dims->data[1];
  input_dims.w = inputs[0]->dims->data[2];
  input_dims.c = filter->dims->data[3];

  cmsis_nn_dims filter_dims;
  filter_dims.n = filter->dims->data[0];
  filter_dims.h = filter->dims->data[1];
  filter_dims.w = filter->dims->data[2];
  filter_dims.c = filter->dims->data[3];

  cmsis_nn_dims output_dims;
  output_dims.n = output->dims->data[0];
  output_dims.h = output->dims->data[1];
  output_dims.w = output->dims->data[2];
  output_dims.c = output->dims->data[3];

  const int num_channels = filter->dims->data[3];
  data->output_zero_point = output->params.zero_point;
  data->output_multiplier = static_cast<int32_t*>(context->AllocatePersistentBuffer(
    context, num_channels * sizeof(int32_t)));
  data->output_shift = static_cast<int32_t*>(context->AllocatePersistentBuffer(
    context, num_channels * sizeof(int32_t)));

  int out_height = output_dims.h;
  int out_width = output_dims.w;
  data->padding = ComputePaddingHeightWidth(
    data->stride_height,  data->stride_width,  data->dilation_height_factor,
    data->dilation_width_factor, input_dims.h, input_dims.w, filter_dims.h, filter_dims.w,
    ConvertPadding(padding_option), &out_height, &out_width);

  PopulateConvolutionQuantizationParams(
    context, inputs[0], filter, bias, output, ConvertActivation(data->activation),
    nullptr, nullptr, &data->output_activation_min, &data->output_activation_max,
    data->output_multiplier, data->output_shift,
    output_dims.c);

  cmsis_nn_conv_params conv_params;
  conv_params.output_offset = output->params.zero_point;
  conv_params.stride.h = data->stride_height;
  conv_params.stride.w = data->stride_width;
  conv_params.dilation.h = data->dilation_height_factor;
  conv_params.dilation.w = data->dilation_width_factor;
  conv_params.padding.h = data->padding.height;
  conv_params.padding.w = data->padding.width;
  conv_params.activation.min = data->output_activation_min;
  conv_params.activation.max = data->output_activation_max;

  int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(
    &conv_params, &input_dims, &filter_dims, &output_dims);

  if (buf_size > 0) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, buf_size, &data->buffer_idx));
  } else {
    data->buffer_idx = -1;
  }

  int32_t scratch_size = output_dims.w * output_dims.c * filter_dims.h;
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
    context, scratch_size, &data->scratch_idx));

  for (int i = 0; i < actual_num_inputs; i++) {
    micro_context->DeallocateTempTfLiteTensor(inputs[i]);
  }
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (data->use_bias) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

inline void GetAllInputTensorDims(const TfLiteContext* context,
                                    const TensorInfo* tensors,
                                    const int num_tensors,
                                    cmsis_nn_dims shapes[kMaxInputNum]) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(tensors != nullptr);
  for (int i = 0; i < num_tensors; ++i) {
    const TfLiteEvalTensor* t = context->GetEvalTensor(context, tensors[i].id);
    RuntimeShape shape = tflite::micro::GetTensorShape(t);
    shapes[i].n = shape.Dims(0);
    shapes[i].h = shape.Dims(1);
    shapes[i].w = shape.Dims(2);
    shapes[i].c = shape.Dims(3);
  }
}

inline void GetAllInputTensorData(const TfLiteContext* context,
                                  const TensorInfo* tensors,
                                  const int num_tensors,
                                  int8_t* data[kMaxInputNum]) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(tensors != nullptr);
  for (int i = 0; i < num_tensors; ++i) {
    TfLiteEvalTensor* tensor = context->GetEvalTensor(context, tensors[i].id);
    if (!tensors[i].fetching) {
      data[i] = tflite::micro::GetTensorData<int8_t>(tensor);
    } else {
      data[i] += ElementCount(*tensor->dims);
    }
  }
}

inline void GetAllInputTensorOffset(const TfLiteContext* context,
                                  const TensorInfo* tensors,
                                  const int num_tensors,
                                  int32_t offset[kMaxInputNum]) {
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(tensors != nullptr);
  for (int i = 0; i < num_tensors; ++i) {
    offset[i] = -tensors[i].zero_point;
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  const int kConvFilterTensor = NumInputs(node) - (data.use_bias + 1);
  const TfLiteEvalTensor* filter =
    tflite::micro::GetEvalInput(context, node, kConvFilterTensor);
  const TfLiteEvalTensor* bias = (data.use_bias)
    ? tflite::micro::GetEvalInput(context, node, kConvFilterTensor + 1)
    : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params = *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));

  cmsis_nn_conv_params conv_params;
  conv_params.output_offset = data.output_zero_point;
  conv_params.dilation.h = data.dilation_height_factor;
  conv_params.dilation.w = data.dilation_width_factor;
  conv_params.stride.h = data.stride_height;
  conv_params.stride.w = data.stride_width;
  conv_params.padding.h = data.padding.height;
  conv_params.padding.w = data.padding.width;
  conv_params.activation.min = data.output_activation_min;
  conv_params.activation.max = data.output_activation_max;

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = const_cast<int32_t*>(data.output_multiplier);
  quant_params.shift = const_cast<int32_t*>(data.output_shift);

  RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  RuntimeShape bias_shape = tflite::micro::GetTensorShape(bias);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);

  // Consistency check.
  TFLITE_DCHECK_LE(conv_params.activation.min, conv_params.activation.max);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  if (tflite::micro::GetOptionalTensorData<int8_t>(bias)) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  cmsis_nn_context ctx;
  ctx.buf = context->GetScratchBuffer(context, data.buffer_idx);
  ctx.size = 0;

  int8_t *output_data = tflite::micro::GetOptionalTensorData<int8_t>(output);

  cmsis_nn_dims inputs_dims[kMaxInputNum];
  int8_t *inputs_data[kMaxInputNum] = { output_data };
  int32_t inputs_offset[kMaxInputNum] = { 0 };

  GetAllInputTensorDims(context, data.inputs, data.num_inputs, inputs_dims);
  GetAllInputTensorData(context, data.inputs, data.num_inputs, inputs_data);
  GetAllInputTensorOffset(context, data.inputs, data.num_inputs, inputs_offset);

  cmsis_nn_dims filter_dims;
  filter_dims.n = output_shape.Dims(0);
  filter_dims.h = filter_shape.Dims(1);
  filter_dims.w = filter_shape.Dims(2);
  filter_dims.c = filter_shape.Dims(3);

  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output_depth;

  cmsis_nn_dims output_dims;
  output_dims.n = output_shape.Dims(0);
  output_dims.h = output_shape.Dims(1);
  output_dims.w = output_shape.Dims(2);
  output_dims.c = output_depth;

  for (int i = 0; i < data.num_inputs; ++i) {
    if (data.inputs[i].fetching) {
      int num_elems = inputs_dims[i].n * inputs_dims[i].h * inputs_dims[i].w * inputs_dims[i].c;
      transfer_module->transfer_in(data.inputs[i].id, (uint8_t*) inputs_data[i], num_elems);
    }
  }

  TFLITE_DCHECK_EQ(
      arm_fused_convolve_s8(
          &ctx, &conv_params, &quant_params, inputs_dims, (const int8_t**) inputs_data, inputs_offset, 
          data.num_inputs, &filter_dims, tflite::micro::GetTensorData<int8_t>(filter), &bias_dims,
          tflite::micro::GetOptionalTensorData<int32_t>(bias), &output_dims,
          output_data, (int8_t*) context->GetScratchBuffer(context, data.scratch_idx)),
      ARM_CMSIS_NN_SUCCESS);

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration* Register_FUSED_CONV_2D_INT8() {
  static TfLiteRegistration r = tflite::micro::RegisterOp(Init, Prepare, Eval);
  return &r;
}

}  // namespace tflite

#endif
