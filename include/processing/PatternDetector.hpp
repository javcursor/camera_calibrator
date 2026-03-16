#pragma once

#include "processing/PatternConfig.hpp"

#include <opencv2/opencv.hpp>

#include <vector>

struct DetectionResult {
  bool found = false;
  std::vector<cv::Point2f> corners;
  std::vector<int> ids; // for ChArUco
};

class PatternDetector {
 public:
  explicit PatternDetector(const PatternConfig& config);

  void setConfig(const PatternConfig& config);
  const PatternConfig& config() const;

  DetectionResult detect(const cv::Mat& gray) const;

 private:
  PatternConfig config_;
};
