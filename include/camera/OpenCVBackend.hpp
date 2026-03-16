#pragma once

#include "camera/ICameraBackend.hpp"

#include <opencv2/opencv.hpp>

class OpenCVBackend : public ICameraBackend {
 public:
  BackendType type() const override { return BackendType::OpenCV; }
  std::string name() const override { return "OpenCV"; }

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
  cv::VideoCapture cap_;
  std::string status_;
  std::string gstreamer_pipeline_;
  bool gstreamer_enabled_ = false;
};
