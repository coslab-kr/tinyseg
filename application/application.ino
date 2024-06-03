#include <Arduino.h>

#include "tinyseg_transfer_module.h"

#define SPILLING_TYPE REMOTE_SPILLING

#if defined(CORE_CM7)

#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "model_data.h"

namespace tflite {

TfLiteRegistration* Register_SPILL();
TfLiteRegistration* Register_FETCHING();
TfLiteRegistration* Register_FUSED_CONV_2D_INT8();

}

namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 250 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

} // namespace

void setup() {
  tflite::InitializeTarget();

  transfer_module = new TensorTransferModule(SPILLING_TYPE);
  transfer_module->init();

  model = tflite::GetModel(g_model_opt_data);

  static tflite::MicroMutableOpResolver<15> micro_op_resolver;
  micro_op_resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
  micro_op_resolver.AddMaxPool2D(tflite::Register_MAX_POOL_2D_INT8());
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddTransposeConv();
  micro_op_resolver.AddConcatenation();
  micro_op_resolver.AddShape();
  micro_op_resolver.AddStridedSlice();
  micro_op_resolver.AddPack();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddSplitV();
  micro_op_resolver.AddCustom("Spill", tflite::Register_SPILL());
  micro_op_resolver.AddCustom("Fetching", tflite::Register_FETCHING());
  micro_op_resolver.AddCustom("FusedConv2D", tflite::Register_FUSED_CONV_2D_INT8());

  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
   MicroPrintf("Failed to allocate tensors");
   return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  delay(5000);
}

void loop() {
  for (int i = 0; i < kNumCols * kNumRows * kNumChannels; i++) {
    input->data.uint8[i] = 0;
  }

  unsigned long start = millis();
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.\n");
  }
  unsigned long end = millis();
  MicroPrintf("Latency: %u", end - start);
  MicroPrintf("Output: %u, %u", output->data.uint8[0], output->data.uint8[1024]);

  delay(5000);
}

#elif defined(CORE_CM4)

void setup() {
  transfer_module = new TensorTransferModule(SPILLING_TYPE);
  transfer_module->init();
}

void loop() {
  while (true) {
    digitalWrite(LED_BUILTIN, LOW);
    delay(5000);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(5000);
  }
}

#endif
