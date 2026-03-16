#pragma once

#include "core/Frame.hpp"

#include <mutex>
#include <vector>

class FrameRingBuffer {
 public:
  explicit FrameRingBuffer(size_t capacity = 4)
      : capacity_(capacity), frames_(capacity) {}

  void push(Frame frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    frames_[write_index_] = std::move(frame);
    write_index_ = (write_index_ + 1) % capacity_;
    has_frame_ = true;
  }

  bool latest(Frame& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_frame_) return false;
    size_t index = (write_index_ + capacity_ - 1) % capacity_;
    out = frames_[index];
    return !out.empty();
  }

  size_t capacity() const { return capacity_; }

 private:
  size_t capacity_ = 4;
  mutable std::mutex mutex_;
  std::vector<Frame> frames_;
  size_t write_index_ = 0;
  bool has_frame_ = false;
};
