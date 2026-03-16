#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

enum class PixelFormat {
  Unknown = 0,
  Mono8,
  BayerRG8,
  BayerBG8,
  BayerGB8,
  BayerGR8,
  RGB8,
  BGR8,
};

struct Frame {
  std::shared_ptr<uint8_t> data;
  size_t data_size = 0;
  int width = 0;
  int height = 0;
  int stride = 0;
  PixelFormat format = PixelFormat::Unknown;
  uint64_t timestamp_ns = 0;
  uint64_t seq = 0;

  bool empty() const { return !data || width <= 0 || height <= 0; }
};

inline std::shared_ptr<uint8_t> makeFrameBuffer(size_t size) {
  return std::shared_ptr<uint8_t>(new uint8_t[size], std::default_delete<uint8_t[]>());
}
