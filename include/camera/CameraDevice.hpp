#pragma once

#include "camera/BackendFactory.hpp"
#include "core/FrameRingBuffer.hpp"

#include <atomic>
#include <mutex>
#include <thread>

enum class CameraState {
  Disconnected,
  Connected,
  Streaming,
  Error,
};

class CameraDevice {
 public:
  CameraDevice();
  ~CameraDevice();

  void setBackend(BackendType type);
  BackendType backendType() const;
  std::string backendName() const;

  std::vector<CameraInfo> listDevices();
  bool connect(const CameraInfo& info);
  void disconnect();

  bool startStreaming();
  void stopStreaming();

  bool latestFrame(Frame& out) const;

  CameraState state() const;
  std::string status() const;
  CameraInfo currentInfo() const;
  double captureFps() const;
  uint64_t capturedFrames() const;
  uint64_t captureFailures() const;

  bool supportsFeatures() const;
  std::vector<FeatureInfo> listFeatures();
  bool getFeatureValue(const FeatureInfo& info, FeatureValue& out);
  bool setFeatureValue(const FeatureInfo& info, const FeatureValue& value);
  bool executeCommand(const FeatureInfo& info);

  bool setBackendOption(const std::string& key, const std::string& value);

 private:
  void captureLoop();
  uint64_t nowNs() const;

  std::unique_ptr<ICameraBackend> backend_;
  BackendType backend_type_ = BackendType::OpenCV;
  CameraInfo current_info_;

  mutable std::mutex backend_mutex_;
  mutable std::mutex state_mutex_;
  CameraState state_ = CameraState::Disconnected;
  std::string status_;

  std::atomic<bool> running_{false};
  std::thread capture_thread_;

  FrameRingBuffer ring_{4};
  uint64_t seq_ = 0;
  std::atomic<double> capture_fps_{0.0};
  std::atomic<uint64_t> captured_frames_{0};
  std::atomic<uint64_t> capture_failures_{0};
};
