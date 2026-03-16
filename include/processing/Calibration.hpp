#pragma once

#include "processing/PatternConfig.hpp"

#include <opencv2/opencv.hpp>

#include <array>
#include <functional>
#include <string>
#include <vector>

struct SampleMetrics {
  float px = 0.0f;
  float py = 0.0f;
  float scale = 0.0f;
  float skew = 0.0f;
};

enum class CameraModel {
  Pinhole,
  Fisheye,
};

enum class PinholeDistortionModel {
  PlumbBob,
  RationalPolynomial,
};

struct CalibrationProgress {
  float x = 0.0f;
  float y = 0.0f;
  float scale = 0.0f;
  float skew = 0.0f;
};

struct ResidualGrid {
  int cols = 0;
  int rows = 0;
  std::vector<float> mean_error;
  std::vector<float> mean_dx;
  std::vector<float> mean_dy;
};

struct CalibrationResult {
  bool valid = false;
  double rms = 0.0;
  CameraModel model = CameraModel::Pinhole;
  PinholeDistortionModel pinhole_distortion = PinholeDistortionModel::PlumbBob;
  cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
  cv::Mat refined_object_points;
  cv::Size image_size;
  bool target_warp_compensation_used = false;
  std::vector<double> per_view_errors;
  std::vector<int> inlier_sample_indices;
  std::vector<int> rejected_sample_indices;
  double mean_reprojection_error = 0.0;
  double median_reprojection_error = 0.0;
  double p95_reprojection_error = 0.0;
  double outlier_threshold = 0.0;
  ResidualGrid residual_grid;
};

struct StereoCalibrationResult {
  bool valid = false;
  double rms = 0.0;
  cv::Mat camera_matrix_left;
  cv::Mat dist_coeffs_left;
  cv::Mat camera_matrix_right;
  cv::Mat dist_coeffs_right;
  cv::Mat rotation;
  cv::Mat translation;
  cv::Mat essential;
  cv::Mat fundamental;
  cv::Mat rectification_left;
  cv::Mat rectification_right;
  cv::Mat projection_left;
  cv::Mat projection_right;
  cv::Mat disparity_to_depth;
  cv::Mat rect_map1_left;
  cv::Mat rect_map2_left;
  cv::Mat rect_map1_right;
  cv::Mat rect_map2_right;
  cv::Rect valid_roi_left;
  cv::Rect valid_roi_right;
  cv::Size image_size;
  double baseline = 0.0;
};

using CalibrationProgressCallback = std::function<void(float progress, const std::string& message)>;

class CalibrationSession {
 public:
  void setPattern(const PatternConfig& config);
  void setCameraModel(CameraModel model);
  CameraModel cameraModel() const { return camera_model_; }
  void setPinholeDistortionModel(PinholeDistortionModel model);
  PinholeDistortionModel pinholeDistortionModel() const { return pinhole_distortion_model_; }
  void setTargetWarpCompensation(bool enabled);
  bool targetWarpCompensation() const { return target_warp_compensation_; }
  void clear();

  bool addSample(const std::vector<cv::Point2f>& corners,
                 const std::vector<int>& ids,
                 const cv::Size& image_size,
                 bool allow_duplicates);

  CalibrationProgress progress() const;
  bool goodEnough() const;
  int sampleCount() const;

  const std::string& lastStatus() const;
  const CalibrationResult& result() const;

  bool calibrate(const cv::Size& image_size,
                 const CalibrationProgressCallback& progress_callback = {});
  bool saveResult(const std::string& path) const;

  SampleMetrics computeMetrics(const std::vector<cv::Point2f>& corners,
                               const std::vector<int>& ids,
                               const cv::Size& image_size) const;

 private:
  struct Sample {
    std::vector<cv::Point2f> corners;
    std::vector<int> ids;
    SampleMetrics metrics;
  };

  std::vector<cv::Point3f> buildObjectPoints() const;
  bool isDistinct(const SampleMetrics& m) const;

  bool getOutsideCorners(const std::vector<cv::Point2f>& corners,
                         const std::vector<int>& ids,
                         std::array<cv::Point2f, 4>& out) const;

  PatternConfig config_;
  CameraModel camera_model_ = CameraModel::Pinhole;
  PinholeDistortionModel pinhole_distortion_model_ = PinholeDistortionModel::PlumbBob;
  bool target_warp_compensation_ = false;
  std::vector<Sample> samples_;
  CalibrationResult result_;
  std::string last_status_;
};

class StereoCalibrationSession {
 public:
  void setPattern(const PatternConfig& config);
  void clear();
  void setExpectedBaseline(double baseline_m);

  bool addSample(const std::vector<cv::Point2f>& left_corners,
                 const std::vector<int>& left_ids,
                 const cv::Size& left_image_size,
                 const std::vector<cv::Point2f>& right_corners,
                 const std::vector<int>& right_ids,
                 const cv::Size& right_image_size,
                 bool allow_duplicates);

  int sampleCount() const;
  const std::string& lastStatus() const;
  const StereoCalibrationResult& result() const;
  double expectedBaseline() const { return expected_baseline_m_; }

  bool calibrate(bool fix_intrinsics);
  bool saveResult(const std::string& path) const;

 private:
  struct StereoSample {
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> left_points;
    std::vector<cv::Point2f> right_points;
    cv::Size image_size;
    SampleMetrics left_metrics;
    SampleMetrics right_metrics;
  };

  struct StereoSignature {
    float left_px = 0.0f;
    float left_py = 0.0f;
    float left_scale = 0.0f;
    float right_px = 0.0f;
    float right_py = 0.0f;
    float right_scale = 0.0f;
  };

  SampleMetrics computeSimpleMetrics(const std::vector<cv::Point2f>& corners,
                                     const cv::Size& image_size) const;
  StereoSignature buildSignature(const SampleMetrics& left, const SampleMetrics& right) const;
  bool isDistinct(const StereoSignature& signature) const;
  bool buildRegularObjectPoints(std::vector<cv::Point3f>& out) const;
  bool buildStereoPoints(const std::vector<cv::Point2f>& left_corners,
                         const std::vector<int>& left_ids,
                         const std::vector<cv::Point2f>& right_corners,
                         const std::vector<int>& right_ids,
                         std::vector<cv::Point3f>& object_points,
                         std::vector<cv::Point2f>& left_points,
                         std::vector<cv::Point2f>& right_points);

  PatternConfig config_;
  std::vector<StereoSample> samples_;
  std::vector<StereoSignature> signatures_;
  StereoCalibrationResult result_;
  std::string last_status_;
  double expected_baseline_m_ = 0.0;
};
