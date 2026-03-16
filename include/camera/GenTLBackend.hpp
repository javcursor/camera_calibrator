#pragma once

#include "camera/ICameraBackend.hpp"

#include <cstdint>
#include <string>
#include <vector>

class GenTLBackend : public ICameraBackend {
 public:
  ~GenTLBackend() override;

  BackendType type() const override { return BackendType::GenTL; }
  std::string name() const override { return "GenTL"; }

  std::vector<CameraInfo> listDevices() override;
  bool open(const CameraInfo& info) override;
  void close() override;
  bool start() override;
  void stop() override;
  bool grab(Frame& out) override;
  bool isOpen() const override;

  bool setOption(const std::string& key, const std::string& value) override;
  std::string status() const override { return status_; }

 private:
  using GenTLHandle = void*;
  using GenTLError = int32_t;

  bool load();
  void unload();
  bool openDataStream();
  bool registerNewBufferEvent();
  bool queryPayloadSize(size_t& payload_size) const;
  bool queryMinAnnouncedBuffers(size_t& min_buffers) const;
  void setStatusFromError(const char* operation, GenTLError err);
  bool queueBuffer(GenTLHandle buffer);

  std::string cti_path_;
  std::string status_;

  GenTLHandle lib_ = nullptr;
  GenTLHandle tl_ = nullptr;
  GenTLHandle if_handle_ = nullptr;
  GenTLHandle dev_ = nullptr;
  GenTLHandle ds_ = nullptr;
  GenTLHandle new_buffer_event_ = nullptr;
  std::vector<GenTLHandle> announced_buffers_;
  size_t payload_size_ = 0;
  bool acquisition_started_ = false;
};
