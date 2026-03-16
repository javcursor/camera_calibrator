#include "camera/CameraDevice.hpp"

#include <chrono>

CameraDevice::CameraDevice() {
  setBackend(availableBackends().front());
}

CameraDevice::~CameraDevice() {
  stopStreaming();
  disconnect();
}

void CameraDevice::setBackend(BackendType type) {
  stopStreaming();
  disconnect();
  {
    std::lock_guard<std::mutex> lock(backend_mutex_);
    backend_ = createBackend(type);
  }
  backend_type_ = type;
}

BackendType CameraDevice::backendType() const {
  return backend_type_;
}

std::string CameraDevice::backendName() const {
  return backend_ ? backend_->name() : "";
}

std::vector<CameraInfo> CameraDevice::listDevices() {
  std::vector<CameraInfo> devices;
  std::string backend_status;
  {
    std::lock_guard<std::mutex> backend_lock(backend_mutex_);
    if (!backend_) return {};
    devices = backend_->listDevices();
    backend_status = backend_->status();
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    status_ = backend_status;
  }
  return devices;
}

bool CameraDevice::connect(const CameraInfo& info) {
  bool opened = false;
  std::string backend_status;
  {
    std::lock_guard<std::mutex> backend_lock(backend_mutex_);
    if (!backend_) return false;
    opened = backend_->open(info);
    backend_status = backend_->status();
  }

  std::lock_guard<std::mutex> state_lock(state_mutex_);
  current_info_ = {};
  if (opened) {
    current_info_ = info;
    state_ = CameraState::Connected;
    status_ = backend_status;
    return true;
  }
  state_ = CameraState::Error;
  status_ = backend_status;
  return false;
}

void CameraDevice::disconnect() {
  {
    std::lock_guard<std::mutex> backend_lock(backend_mutex_);
    if (!backend_) return;
    backend_->close();
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_ = CameraState::Disconnected;
  status_.clear();
  current_info_ = {};
}

bool CameraDevice::startStreaming() {
  {
    std::lock_guard<std::mutex> backend_lock(backend_mutex_);
    if (!backend_ || !backend_->isOpen()) return false;
  }
  if (running_) return true;
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
  capture_fps_.store(0.0, std::memory_order_relaxed);
  captured_frames_.store(0, std::memory_order_relaxed);
  capture_failures_.store(0, std::memory_order_relaxed);
  running_ = true;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_ = CameraState::Streaming;
    status_ = "Starting stream...";
  }
  capture_thread_ = std::thread(&CameraDevice::captureLoop, this);
  return true;
}

void CameraDevice::stopStreaming() {
  running_ = false;
  {
    std::lock_guard<std::mutex> backend_lock(backend_mutex_);
    if (backend_) backend_->stop();
  }
  if (capture_thread_.joinable()) capture_thread_.join();
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (state_ == CameraState::Streaming) {
    state_ = CameraState::Connected;
  }
  capture_fps_.store(0.0, std::memory_order_relaxed);
}

bool CameraDevice::latestFrame(Frame& out) const {
  return ring_.latest(out);
}

CameraState CameraDevice::state() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_;
}

std::string CameraDevice::status() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return status_;
}

CameraInfo CameraDevice::currentInfo() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return current_info_;
}

double CameraDevice::captureFps() const {
  return capture_fps_.load(std::memory_order_relaxed);
}

uint64_t CameraDevice::capturedFrames() const {
  return captured_frames_.load(std::memory_order_relaxed);
}

uint64_t CameraDevice::captureFailures() const {
  return capture_failures_.load(std::memory_order_relaxed);
}

bool CameraDevice::supportsFeatures() const {
  return backend_ && backend_->supportsFeatures();
}

std::vector<FeatureInfo> CameraDevice::listFeatures() {
  std::lock_guard<std::mutex> backend_lock(backend_mutex_);
  if (!backend_) return {};
  return backend_->listFeatures();
}

bool CameraDevice::getFeatureValue(const FeatureInfo& info, FeatureValue& out) {
  std::lock_guard<std::mutex> backend_lock(backend_mutex_);
  if (!backend_) return false;
  return backend_->getFeatureValue(info, out);
}

bool CameraDevice::setFeatureValue(const FeatureInfo& info, const FeatureValue& value) {
  std::lock_guard<std::mutex> backend_lock(backend_mutex_);
  if (!backend_) return false;
  return backend_->setFeatureValue(info, value);
}

bool CameraDevice::executeCommand(const FeatureInfo& info) {
  std::lock_guard<std::mutex> backend_lock(backend_mutex_);
  if (!backend_) return false;
  return backend_->executeCommand(info);
}

bool CameraDevice::setBackendOption(const std::string& key, const std::string& value) {
  std::lock_guard<std::mutex> backend_lock(backend_mutex_);
  if (!backend_) return false;
  return backend_->setOption(key, value);
}

void CameraDevice::captureLoop() {
  bool started = false;
  std::string backend_status;
  {
    std::lock_guard<std::mutex> backend_lock(backend_mutex_);
    if (!backend_) return;
    started = backend_->start();
    backend_status = backend_->status();
  }
  if (!started) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_ = CameraState::Connected;
    status_ = backend_status;
    running_ = false;
    return;
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_ = CameraState::Streaming;
    status_ = backend_status.empty() ? "Streaming" : backend_status;
  }

  auto fps_window_start = std::chrono::steady_clock::now();
  uint64_t frames_in_window = 0;
  int failures = 0;
  while (running_) {
    Frame frame;
    bool grabbed = false;
    {
      std::lock_guard<std::mutex> backend_lock(backend_mutex_);
      grabbed = backend_ && backend_->grab(frame);
    }
    if (grabbed) {
      if (frame.timestamp_ns == 0) {
        frame.timestamp_ns = nowNs();
      }
      frame.seq = ++seq_;
      ring_.push(std::move(frame));
      captured_frames_.fetch_add(1, std::memory_order_relaxed);
      ++frames_in_window;
      const auto now = std::chrono::steady_clock::now();
      const auto elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_window_start).count();
      if (elapsed_ms >= 500) {
        const double fps = (static_cast<double>(frames_in_window) * 1000.0) /
                           static_cast<double>(elapsed_ms);
        capture_fps_.store(fps, std::memory_order_relaxed);
        fps_window_start = now;
        frames_in_window = 0;
      }
      failures = 0;
    } else {
      ++failures;
      capture_failures_.fetch_add(1, std::memory_order_relaxed);
      const auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::seconds>(now - fps_window_start).count() >= 1 &&
          frames_in_window == 0) {
        capture_fps_.store(0.0, std::memory_order_relaxed);
      }
      if (failures == 1 || (failures % 60) == 0) {
        std::string wait_status = "Waiting for frames...";
        {
          std::lock_guard<std::mutex> backend_lock(backend_mutex_);
          if (backend_ && !backend_->status().empty()) {
            wait_status = backend_->status();
          }
        }
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (state_ == CameraState::Streaming) {
          status_ = wait_status;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  std::lock_guard<std::mutex> backend_lock(backend_mutex_);
  if (backend_) backend_->stop();
}

uint64_t CameraDevice::nowNs() const {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
