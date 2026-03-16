#pragma once

#include <imgui.h>
#include <opencv2/opencv.hpp>

class ImageTexture {
 public:
  ~ImageTexture();

  void update(const cv::Mat& rgba);
  ImTextureID id() const;
  int width() const;
  int height() const;
  void release();

 private:
  unsigned int tex_ = 0;
  int width_ = 0;
  int height_ = 0;
};
