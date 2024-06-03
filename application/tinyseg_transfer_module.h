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

#pragma once

#include <Arduino.h>
#include <map>

#include "tinyseg_tensor_storage.h"

using namespace mbed;

#define INTERNAL_SPILLING 0
#define EXTERNAL_SPILLING 1
#define REMOTE_SPILLING 2

#define NUM_SECTIONS 4

struct TensorInfo {
  int pos;
  int size;
  int cur;
};

struct CompressInfo {
  uint8_t val;
  int unit;
  int offset[NUM_SECTIONS + 1];
};

class TensorTransferModule {
public:
  TensorTransferModule(int storage_type);
  void init();

  void transfer_in(int id, uint8_t* buffer, int size);
  void transfer_out(int id, uint8_t* buffer, int size);

private:

  int compress(int id, uint8_t* buffer, int size);
  void decompress(int id, uint8_t* buffer, int size);

  TensorStorage* storage;
  std::map<int, TensorInfo> tensor_map;
  std::map<int, CompressInfo> compress_map;

  int type;
};

extern TensorTransferModule* transfer_module;
