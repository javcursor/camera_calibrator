#include "ui/ImageTexture.hpp"

#include <GLFW/glfw3.h>

ImageTexture::~ImageTexture() {
  release();
}

void ImageTexture::update(const cv::Mat& rgba) {
  if (rgba.empty()) return;
  if (tex_ == 0) {
    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }

  glBindTexture(GL_TEXTURE_2D, tex_);
  if (width_ != rgba.cols || height_ != rgba.rows) {
    width_ = rgba.cols;
    height_ = rgba.rows;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_,
                    GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
  }
}

ImTextureID ImageTexture::id() const {
  return (ImTextureID)(intptr_t)tex_;
}

int ImageTexture::width() const {
  return width_;
}

int ImageTexture::height() const {
  return height_;
}

void ImageTexture::release() {
  if (tex_ != 0) {
    glDeleteTextures(1, &tex_);
    tex_ = 0;
  }
}
