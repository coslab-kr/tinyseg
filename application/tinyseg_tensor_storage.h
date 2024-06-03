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

#include <QSPIFBlockDevice.h>
#include <FlashIAP.h>
#include <FlashIAPBlockDevice.h>

using namespace mbed;

#define SWAP_AREA 128 * 1024

//#define USE_RPC

class TensorStorage {
public:
  virtual void init() = 0;
  virtual void read(int pos, uint8_t *buffer, int size) = 0;
  virtual void write(int pos, const uint8_t *buffer, int size) = 0;
  virtual void erase(int pos, int size) = 0;
  virtual size_t align(size_t size) = 0;
};

struct FlashIAPLimits {
  size_t flash_size;
  uint32_t start_address;
  int available_size;
};

class InternalTensorStorage: public TensorStorage {
public:
  InternalTensorStorage();
  void init();
  void read(int pos, uint8_t *buffer, int size);
  void write(int pos, const uint8_t *buffer, int size);
  void erase(int pos, int size);
  size_t align(size_t size);

private:
  FlashIAPLimits get_flash_info();
  void _erase();

  FlashIAPBlockDevice *blockDevice;
  uint64_t eraseBlockSize;
  uint64_t programBlockSize;
  int erasePos;
  int eraseSize;
};

class ExternalTensorStorage: public TensorStorage {
public:
  ExternalTensorStorage();
  void init();
  void read(int pos, uint8_t* buffer, int size);
  void write(int pos, const uint8_t* buffer, int size);
  void erase(int pos, int size);
  size_t align(size_t size);

private:
  void _erase();

  QSPIFBlockDevice *blockDevice;
  uint64_t eraseBlockSize;
  uint64_t programBlockSize;
  int erasePos;
  int eraseSize;
};

class RemoteTensorStorage: public TensorStorage {
public:
  RemoteTensorStorage();
  void init();
  void read(int pos, uint8_t* buffer, int size);
  void write(int pos, const uint8_t* buffer, int size);
  void erase(int pos, int size);
  size_t align(size_t size);

private:
  char ini_code[11];
  char fin_code[11];
};
