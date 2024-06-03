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

#include <Arduino.h>
#include "mbed.h"

#include "tinyseg_tensor_storage.h"

static rtos::Thread *eraseThread = nullptr;

InternalTensorStorage::InternalTensorStorage(){
  this->blockDevice = nullptr;
  this->eraseBlockSize = 0;
  this->programBlockSize = 0;
  this->erasePos = 0;
  this->eraseSize = 0;
}

void InternalTensorStorage::init() {
  auto [flashSize, startAddress, iapSize] = this->get_flash_info();

  this->blockDevice = new FlashIAPBlockDevice(startAddress, iapSize);
  this->blockDevice->init();
  
  eraseBlockSize = blockDevice->get_erase_size();
  programBlockSize = blockDevice->get_program_size();

  blockDevice->erase(0, SWAP_AREA);
}

FlashIAPLimits InternalTensorStorage::get_flash_info()
{
  auto align_down = [](uint64_t val, uint64_t size) {
    return (((val) / size)) * size;
  };
  auto align_up = [](uint32_t val, uint32_t size) {
    return (((val - 1) / size) + 1) * size;
  };

  size_t flash_size;
  uint32_t flash_start_address;
  uint32_t start_address;
  FlashIAP flash;

  auto result = flash.init();
  if (result != 0)
    return { };

  int sector_size = flash.get_sector_size(FLASHIAP_APP_ROM_END_ADDR);
  start_address = align_up(FLASHIAP_APP_ROM_END_ADDR, sector_size);
  flash_start_address = flash.get_flash_start();
  flash_size = flash.get_flash_size();

  result = flash.deinit();

  int available_size = flash_start_address + flash_size - start_address;
  if (available_size % (sector_size * 2)) {
    available_size = align_down(available_size, sector_size * 2);
  }

  return { flash_size, start_address, available_size };
}

void InternalTensorStorage::read(int pos, uint8_t* buffer, int size) {
  const unsigned int block = pos / this->programBlockSize;
  this->blockDevice->read(buffer, block, size);
}

void InternalTensorStorage::write(int pos, const uint8_t* buffer, int size) {
  const unsigned int block = pos / this->programBlockSize;
  const unsigned int requiredProgramBlocks = ceil(size / (float) this->programBlockSize);
  this->blockDevice->program(buffer, block, requiredProgramBlocks * this->programBlockSize);
}

void InternalTensorStorage::erase(int pos, int size) {
  this->erasePos = pos;
  this->eraseSize = size;
#if defined(USE_RPC)
  eraseThread = new rtos::Thread(osPriorityNormal);
  eraseThread->start(mbed::callback(this, &InternalTensorStorage::_erase));
#else
  this->_erase();
#endif
}

void InternalTensorStorage::_erase() {
  const unsigned int block = this->erasePos / this->programBlockSize;
  const unsigned int requiredEraseBlocks = ceil(this->eraseSize / (float) this->eraseBlockSize);
  this->blockDevice->erase(block, requiredEraseBlocks * this->eraseBlockSize);
}

size_t InternalTensorStorage::align(size_t size) {
  size_t multiple = std::max(this->programBlockSize, this->eraseBlockSize);
  return ((size + multiple - 1) / multiple) * multiple;
}

ExternalTensorStorage::ExternalTensorStorage(){
  this->blockDevice = nullptr;
  this->eraseBlockSize = 0;
  this->programBlockSize = 0;
  this->erasePos = 0;
  this->eraseSize = 0;
}

void ExternalTensorStorage::init() {
  this->blockDevice = new QSPIFBlockDevice;
  this->blockDevice->init();
  
  this->eraseBlockSize = blockDevice->get_erase_size();
  this->programBlockSize = blockDevice->get_program_size();

  this->blockDevice->erase(0, SWAP_AREA);
}

void ExternalTensorStorage::read(int pos, uint8_t* buffer, int size) {
  const unsigned int block = pos / this->programBlockSize;
  this->blockDevice->read(buffer, block, size);
}

void ExternalTensorStorage::write(int pos, const uint8_t* buffer, int size) {
  const unsigned int block = pos / this->programBlockSize;
  const unsigned int requiredProgramBlocks = ceil(size / (float) this->programBlockSize);
  this->blockDevice->program(buffer, block, requiredProgramBlocks * this->programBlockSize);
}

void ExternalTensorStorage::erase(int pos, int size) {
  this->erasePos = pos;
  this->eraseSize = size;
#if defined(USE_RPC)
  eraseThread = new rtos::Thread(osPriorityNormal);
  eraseThread->start(mbed::callback(this, &ExternalTensorStorage::_erase));
#else
  this->_erase();
#endif
}

void ExternalTensorStorage::_erase() {
  const unsigned int block = this->erasePos / this->programBlockSize;
  const unsigned int requiredEraseBlocks = ceil(this->eraseSize / (float) this->eraseBlockSize);
  this->blockDevice->erase(block, requiredEraseBlocks * this->eraseBlockSize);
}

size_t ExternalTensorStorage::align(size_t size) {
  size_t multiple = std::max(this->programBlockSize, this->eraseBlockSize);
  return ((size + multiple - 1) / multiple) * multiple;
}

RemoteTensorStorage::RemoteTensorStorage() {
  strncpy(this->ini_code, "TINYSEG_INI", sizeof(this->ini_code));
  strncpy(this->fin_code, "TINYSEG_FIN", sizeof(this->fin_code));
}

void RemoteTensorStorage::init() {
  Serial.begin(115200);
}

void RemoteTensorStorage::read(int pos, uint8_t* buffer, int size) {
  Serial.write(this->ini_code, 11);
  Serial.write('r');
  Serial.write((uint8_t*)&pos, 4);
  Serial.write((uint8_t*)&size, 4);
  Serial.readBytes(buffer, size);
  Serial.write(this->fin_code, 11);
}

void RemoteTensorStorage::write(int pos, const uint8_t* buffer, int size) {
  Serial.write(this->ini_code, 11);
  Serial.write('w');
  Serial.write((uint8_t*)&pos, 4);
  Serial.write((uint8_t*)&size, 4);
  Serial.write(buffer, size);
  Serial.write(this->fin_code, 11);
}

void RemoteTensorStorage::erase(int pos, int size) {
  // Do nothing
}

size_t RemoteTensorStorage::align(size_t size) {
  return size;
}
