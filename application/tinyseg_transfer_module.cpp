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

#include "SerialRPC.h"

#include "tinyseg_transfer_module.h"

#if defined(USE_RPC)
#include "RPC.h"
#endif

TensorTransferModule *transfer_module = nullptr;

#if defined(CORE_CM7)

TensorTransferModule::TensorTransferModule(int storage_type) {
  this->type = storage_type;

  switch(storage_type) {
#if !defined(USE_RPC)
    case INTERNAL_SPILLING:
      this->storage = new InternalTensorStorage();
      break;
    case EXTERNAL_SPILLING:
      this->storage = new ExternalTensorStorage();
      break;
#endif
    case REMOTE_SPILLING:
      this->storage = new RemoteTensorStorage();
      break;
    default:
      this->storage = nullptr;
  }}

void TensorTransferModule::init() {
#if defined(USE_RPC)
  if (this->type == INTERNAL_SPILLING || this->type == EXTERNAL_SPILLING) {
    RPC.begin();
    return;
  }
#endif

  if (this->storage != nullptr) {
    this->storage->init();
  }
}

void TensorTransferModule::transfer_in(int id, uint8_t* buffer, int size) {
#if defined(USE_RPC)
  if (this->type == INTERNAL_SPILLING || this->type == EXTERNAL_SPILLING) {
    SCB_InvalidateDCache_by_Addr((void*) buffer, size);
    RPC.call("TransferIn", id, (uint32_t) buffer, size);
    return;
  }
#endif

  TensorInfo &tensor_info = this->tensor_map[id];
  CompressInfo &compress_info = this->compress_map[id];

  int start = tensor_info.cur / compress_info.unit;
  int end = start + size / compress_info.unit;

  int compressed_size = compress_info.offset[end] - compress_info.offset[start];
  this->storage->read(tensor_info.pos + compress_info.offset[start], buffer, compressed_size);
  this->decompress(id, buffer, size);

  tensor_info.cur += size;
  if (tensor_info.cur >= tensor_info.size) {
    this->storage->erase(tensor_info.pos, compress_info.offset[NUM_SECTIONS]);
  }
}

void TensorTransferModule::transfer_out(int id, uint8_t* buffer, int size) {
#if defined(USE_RPC)
  if (this->type == INTERNAL_SPILLING || this->type == EXTERNAL_SPILLING) {
    SCB_CleanDCache_by_Addr((void*)const_cast<uint8_t*>(buffer), size);
    RPC.call("TransferOut", id, (uint32_t) buffer, size);
    return;
  }
#endif

  static int offset = 0;
  int compressed_size = this->compress(id, buffer, size);
  if (this->tensor_map.find(id) == this->tensor_map.end()) {
    this->tensor_map[id].pos = offset;
    this->tensor_map[id].size = size;

    offset += this->storage->align(compressed_size);
  }
  this->storage->write(this->tensor_map[id].pos, buffer, compressed_size);

  this->tensor_map[id].cur = 0;
}

#else

void TransferIn(int id, uint32_t addr, int size) {
  uint8_t* buffer = (uint8_t*) addr;
  transfer_module->transfer_in(id, buffer, size);
}

void TransferOut(int id, uint32_t addr, int size) {
  uint8_t* buffer = (uint8_t*) addr;
  transfer_module->transfer_out(id, buffer, size);
}

TensorTransferModule::TensorTransferModule(int storage_type) {
  this->type = storage_type;

  switch(storage_type) {
    case INTERNAL_SPILLING:
      this->storage = new InternalTensorStorage();
      break;
    case EXTERNAL_SPILLING:
      this->storage = new ExternalTensorStorage();
      break;
    default:
      this->storage = nullptr;
  }
}

void TensorTransferModule::init() {
  if (this->storage != nullptr) {
    RPC.begin();
    RPC.bind("TransferIn", TransferIn);
    RPC.bind("TransferOut", TransferOut);
    this->storage->init();
  }
}

void TensorTransferModule::transfer_in(int id, uint8_t* buffer, int size) {
  TensorInfo &tensor_info = this->tensor_map[id];
  CompressInfo &compress_info = this->compress_map[id];

  int start = tensor_info.cur / compress_info.unit;
  int end = start + size / compress_info.unit;

  int compressed_size = compress_info.offset[end] - compress_info.offset[start];
  this->storage->read(tensor_info.pos + compress_info.offset[start], buffer, compressed_size);
  this->decompress(id, buffer, size);

  tensor_info.cur += size;
  if (tensor_info.cur >= tensor_info.size) {
    this->storage->erase(tensor_info.pos, compress_info.offset[NUM_SECTIONS]);
  }
}

void TensorTransferModule::transfer_out(int id, uint8_t* buffer, int size) {
  static int offset = 0;

  int compressed_size = this->compress(id, buffer, size);
  if (this->tensor_map.find(id) == this->tensor_map.end()) {
    this->tensor_map[id].pos = offset;
    this->tensor_map[id].size = size;

    offset += this->storage->align(compressed_size);
  }

  this->storage->write(this->tensor_map[id].pos, buffer, compressed_size);
  this->tensor_map[id].cur = 0;
}

#endif

static uint8_t find_max_value(const uint8_t* buffer, const int size) {
  uint32_t count[256] = { 0 };

  for (int i = 0; i < 256; i++) {
    count[i] = 0;
  }

  for (int i = 0; i < size; i++) {
    uint8_t value = buffer[i];
    count[value] += 1;
  }

  int maxCount = 0;
  uint8_t maxVal = 0;

  for (int i = 0; i < 256; i++) {
    if (count[i] > maxCount) {
      maxCount = count[i];
      maxVal = (uint8_t) i;
    }
  }

  return maxVal;
}

int TensorTransferModule::compress(int id, uint8_t* buffer, int size) {
  const uint8_t val = find_max_value(buffer, size);
  const int section_size = size / NUM_SECTIONS;

  this->compress_map[id].val = val;
  this->compress_map[id].unit = section_size;

  int num = 0;
  uint8_t temp[8] = { 0 };

  for (int i = 0; i < size; i += 8) {
    for (int j = 0; j < 8; j++) {
      temp[j] = buffer[i + j];
    }

    if (i % section_size == 0) {
      this->compress_map[id].offset[i / section_size] = num;
    }

    uint8_t mask = 0;
    for (int j = 0; j < 8; j++) {
      if (temp[j] != val) {
        assert(num < i + 8);
        buffer[num++] = temp[j];
        mask |= 1 << j;
      }
    }
    buffer[num++] = mask;
  }

  this->compress_map[id].offset[NUM_SECTIONS] = num;
  return num;
}

void TensorTransferModule::decompress(int id, uint8_t* buffer, int size) {
  TensorInfo &tensor_info = this->tensor_map[id];
  CompressInfo &compress_info = this->compress_map[id];

  assert(size % compress_info.unit == 0);
  int start = tensor_info.cur / compress_info.unit;
  int end = start + size / compress_info.unit;

  int num = compress_info.offset[end] - compress_info.offset[start];
  for (int i = size - 1; i >= 0; i -= 8) {
    uint8_t mask = buffer[--num];

    for (int j = 7; j >= 0; j--) {
      if (mask & (1 << j)) {
        buffer[i-7+j] = buffer[--num];
      } else {
        buffer[i-7+j] = compress_info.val;
      }
    }
  }

  return;
}
