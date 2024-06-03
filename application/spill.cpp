/* Copyright 2024 COS Lab at Kyung Hee University. All Rights Reserved.

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
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"

#include "tinyseg_transfer_module.h"

namespace tflite {

namespace {

const int kInputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(NumInputs(node) == 1);
  TFLITE_DCHECK(NumOutputs(node) == 0);

  MicroContext* micro_context = GetMicroContext(context);
  TF_LITE_ENSURE(context, node->custom_initial_data_size == 1);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input_tensor =
      micro::GetEvalInput(context, node, kInputTensor);
  TFLITE_DCHECK(input_tensor != nullptr);
 
  size_t elem_size = 1;
  TfLiteStatus ret = TfLiteTypeSizeOf(input_tensor->type, &elem_size);

  const uint8_t spill_id = ((uint8_t*) node->custom_initial_data)[0];
  uint8_t* spill_data = const_cast<uint8_t*>(tflite::micro::GetTensorData<uint8_t>(input_tensor));
  const int spill_size = ElementCount(*input_tensor->dims) * elem_size;

  transfer_module->transfer_out(spill_id, spill_data, spill_size); 
  return kTfLiteOk;
}

}  // namespace.

TfLiteRegistration* Register_SPILL() {
  static TfLiteRegistration r = tflite::micro::RegisterOp(nullptr, Prepare, Eval);
  return &r;
}

}

#endif
