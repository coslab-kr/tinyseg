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

#ifndef TINYSEG_APPLICATION_MODEL_DATA_H_
#define TINYSEG_APPLICATION_MODEL_DATA_H_

constexpr int kNumCols = 120;
constexpr int kNumRows = 80;
constexpr int kNumChannels = 3;

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

// extern const unsigned char g_model_data[];
// extern const int g_model_data_len;

extern const unsigned char g_model_opt_data[];
extern const int g_model_opt_data_len;

#endif  // TINYSEG_APPLICATION_MODEL_DATA_H_
