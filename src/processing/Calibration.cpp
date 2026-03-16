#include "processing/Calibration.hpp"

#ifdef HAVE_OPENCV_ARUCO
#include <opencv2/aruco/charuco.hpp>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace {
float clamp01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

float calculate_skew(const std::array<cv::Point2f, 4>& corners) {
  const cv::Point2f& up_left = corners[0];
  const cv::Point2f& up_right = corners[1];
  const cv::Point2f& down_right = corners[2];

  auto angle = [](const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c) {
    cv::Point2f ab = a - b;
    cv::Point2f cb = c - b;
    double denom = cv::norm(ab) * cv::norm(cb);
    if (denom <= 1e-6) return 0.0;
    double cosang = std::clamp((ab.dot(cb)) / denom, -1.0, 1.0);
    return std::acos(cosang);
  };

  double skew = std::min(1.0, 2.0 * std::abs((CV_PI * 0.5) - angle(up_left, up_right, down_right)));
  return clamp01(static_cast<float>(skew));
}

float calculate_area(const std::array<cv::Point2f, 4>& corners) {
  const cv::Point2f& up_left = corners[0];
  const cv::Point2f& up_right = corners[1];
  const cv::Point2f& down_right = corners[2];
  const cv::Point2f& down_left = corners[3];

  cv::Point2f a = up_right - up_left;
  cv::Point2f b = down_right - up_right;
  cv::Point2f c = down_left - down_right;
  cv::Point2f p = b + c;
  cv::Point2f q = a + b;
  return std::abs(p.x * q.y - p.y * q.x) / 2.0f;
}

const char* cameraModelName(CameraModel model) {
  switch (model) {
    case CameraModel::Pinhole:
      return "pinhole";
    case CameraModel::Fisheye:
      return "fisheye";
    default:
      return "unknown";
  }
}

const char* pinholeModelName(PinholeDistortionModel model) {
  switch (model) {
    case PinholeDistortionModel::PlumbBob:
      return "plumb_bob";
    case PinholeDistortionModel::RationalPolynomial:
      return "rational_polynomial";
    default:
      return "plumb_bob";
  }
}

double medianOf(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  if ((n % 2U) == 0U) {
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
  }
  return values[n / 2U];
}

double percentileOf(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double pos = std::clamp(p, 0.0, 1.0) * static_cast<double>(values.size() - 1U);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) return values[lo];
  const double w = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - w) + values[hi] * w;
}

double meanOf(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  double sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

void reportCalibrationProgress(const CalibrationProgressCallback& callback,
                               float progress,
                               const std::string& message) {
  if (!callback) return;
  callback(clamp01(progress), message);
}

void appendIntSequence(cv::FileStorage& fs, const char* key, const std::vector<int>& values) {
  fs << key << "[";
  for (int v : values) fs << v;
  fs << "]";
}

void appendDoubleSequence(cv::FileStorage& fs, const char* key, const std::vector<double>& values) {
  fs << key << "[";
  for (double v : values) fs << v;
  fs << "]";
}

void appendFloatSequence(cv::FileStorage& fs, const char* key, const std::vector<float>& values) {
  fs << key << "[";
  for (float v : values) fs << v;
  fs << "]";
}

struct PreparedMonoSample {
  int sample_index = -1;
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
};

struct SolveOutput {
  bool ok = false;
  std::string error;
  cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  cv::Mat refined_object_points;
  double rms = 0.0;
};

bool projectPointsForModel(CameraModel model,
                           const std::vector<cv::Point3f>& object_points,
                           const cv::Mat& rvec,
                           const cv::Mat& tvec,
                           const cv::Mat& camera_matrix,
                           const cv::Mat& dist_coeffs,
                           std::vector<cv::Point2f>& projected) {
  try {
    if (model == CameraModel::Fisheye) {
      cv::fisheye::projectPoints(object_points, projected, rvec, tvec, camera_matrix, dist_coeffs);
    } else {
      cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected);
    }
  } catch (const cv::Exception&) {
    return false;
  }
  return projected.size() == object_points.size();
}

std::vector<double> computeViewErrors(const std::vector<PreparedMonoSample>& prepared,
                                      const std::vector<int>& active_indices,
                                      const std::vector<cv::Mat>& rvecs,
                                      const std::vector<cv::Mat>& tvecs,
                                      CameraModel model,
                                      const cv::Mat& camera_matrix,
                                      const cv::Mat& dist_coeffs,
                                      ResidualGrid* residual_grid,
                                      const cv::Size& image_size) {
  std::vector<double> per_view_errors;
  per_view_errors.reserve(active_indices.size());

  const bool compute_grid = (residual_grid != nullptr && image_size.width > 0 && image_size.height > 0);
  std::vector<double> sum_err;
  std::vector<double> sum_dx;
  std::vector<double> sum_dy;
  std::vector<int> counts;
  int grid_cols = 0;
  int grid_rows = 0;
  if (compute_grid) {
    grid_cols = residual_grid->cols;
    grid_rows = residual_grid->rows;
    const int cell_count = grid_cols * grid_rows;
    if (cell_count > 0) {
      sum_err.assign(static_cast<size_t>(cell_count), 0.0);
      sum_dx.assign(static_cast<size_t>(cell_count), 0.0);
      sum_dy.assign(static_cast<size_t>(cell_count), 0.0);
      counts.assign(static_cast<size_t>(cell_count), 0);
    }
  }

  for (size_t i = 0; i < active_indices.size(); ++i) {
    const int prepared_idx = active_indices[i];
    if (prepared_idx < 0 || prepared_idx >= static_cast<int>(prepared.size()) ||
        i >= rvecs.size() || i >= tvecs.size()) {
      per_view_errors.push_back(0.0);
      continue;
    }

    const auto& sample = prepared[static_cast<size_t>(prepared_idx)];
    std::vector<cv::Point2f> projected;
    if (!projectPointsForModel(model, sample.object_points,
                               rvecs[i], tvecs[i],
                               camera_matrix, dist_coeffs,
                               projected)) {
      per_view_errors.push_back(0.0);
      continue;
    }

    double sq_sum = 0.0;
    const size_t n = std::min(projected.size(), sample.image_points.size());
    for (size_t j = 0; j < n; ++j) {
      const cv::Point2f diff = sample.image_points[j] - projected[j];
      const double dx = static_cast<double>(diff.x);
      const double dy = static_cast<double>(diff.y);
      const double e = std::sqrt(dx * dx + dy * dy);
      sq_sum += (dx * dx + dy * dy);

      if (!counts.empty()) {
        int cx = static_cast<int>((sample.image_points[j].x / static_cast<float>(image_size.width)) *
                                  static_cast<float>(grid_cols));
        int cy = static_cast<int>((sample.image_points[j].y / static_cast<float>(image_size.height)) *
                                  static_cast<float>(grid_rows));
        cx = std::clamp(cx, 0, grid_cols - 1);
        cy = std::clamp(cy, 0, grid_rows - 1);
        const size_t cell = static_cast<size_t>(cy * grid_cols + cx);
        sum_err[cell] += e;
        sum_dx[cell] += dx;
        sum_dy[cell] += dy;
        counts[cell] += 1;
      }
    }

    const double rms = (n > 0) ? std::sqrt(sq_sum / static_cast<double>(n)) : 0.0;
    per_view_errors.push_back(rms);
  }

  if (residual_grid && !counts.empty()) {
    residual_grid->mean_error.assign(sum_err.size(), 0.0f);
    residual_grid->mean_dx.assign(sum_dx.size(), 0.0f);
    residual_grid->mean_dy.assign(sum_dy.size(), 0.0f);
    for (size_t i = 0; i < counts.size(); ++i) {
      if (counts[i] <= 0) continue;
      const double denom = static_cast<double>(counts[i]);
      residual_grid->mean_error[i] = static_cast<float>(sum_err[i] / denom);
      residual_grid->mean_dx[i] = static_cast<float>(sum_dx[i] / denom);
      residual_grid->mean_dy[i] = static_cast<float>(sum_dy[i] / denom);
    }
  }

  return per_view_errors;
}

bool hasSameObjectLayout(const std::vector<PreparedMonoSample>& prepared,
                         const std::vector<int>& active_indices) {
  if (active_indices.empty()) return false;
  const auto& ref = prepared[static_cast<size_t>(active_indices[0])].object_points;
  for (size_t i = 1; i < active_indices.size(); ++i) {
    const auto& cur = prepared[static_cast<size_t>(active_indices[i])].object_points;
    if (cur.size() != ref.size()) return false;
    for (size_t k = 0; k < cur.size(); ++k) {
      if (cv::norm(cur[k] - ref[k]) > 1e-9f) return false;
    }
  }
  return true;
}

}  // namespace

void CalibrationSession::setPattern(const PatternConfig& config) {
  if (config_.type != config.type || config_.board_size != config.board_size ||
      std::abs(config_.square_size - config.square_size) > 1e-6f ||
      std::abs(config_.marker_size - config.marker_size) > 1e-6f ||
      config_.aruco_dictionary != config.aruco_dictionary) {
    config_ = config;
    clear();
  }
}

void CalibrationSession::setCameraModel(CameraModel model) {
  if (camera_model_ == model) return;
  camera_model_ = model;
  result_ = CalibrationResult{};
  result_.model = camera_model_;
  result_.pinhole_distortion = pinhole_distortion_model_;
  if (!samples_.empty()) {
    last_status_ = "Modelo de camara actualizado";
  }
}

void CalibrationSession::setPinholeDistortionModel(PinholeDistortionModel model) {
  if (pinhole_distortion_model_ == model) return;
  pinhole_distortion_model_ = model;
  result_ = CalibrationResult{};
  result_.model = camera_model_;
  result_.pinhole_distortion = pinhole_distortion_model_;
  if (!samples_.empty()) {
    last_status_ = "Modelo de distorsion actualizado";
  }
}

void CalibrationSession::setTargetWarpCompensation(bool enabled) {
  if (target_warp_compensation_ == enabled) return;
  target_warp_compensation_ = enabled;
  if (!samples_.empty()) {
    last_status_ = enabled ? "Compensacion de warp activada" : "Compensacion de warp desactivada";
  }
}

void CalibrationSession::clear() {
  samples_.clear();
  result_ = CalibrationResult{};
  result_.model = camera_model_;
  result_.pinhole_distortion = pinhole_distortion_model_;
  last_status_.clear();
}

bool CalibrationSession::addSample(const std::vector<cv::Point2f>& corners,
                                   const std::vector<int>& ids,
                                   const cv::Size& image_size,
                                   bool allow_duplicates) {
  if (corners.empty()) {
    last_status_ = "No se detectaron puntos";
    return false;
  }

  SampleMetrics metrics = computeMetrics(corners, ids, image_size);
  if (!allow_duplicates && !isDistinct(metrics)) {
    last_status_ = "Muestra similar, no agregada";
    return false;
  }

  Sample sample;
  sample.corners = corners;
  sample.ids = ids;
  sample.metrics = metrics;
  samples_.push_back(std::move(sample));
  last_status_ = "Muestra agregada";
  return true;
}

CalibrationProgress CalibrationSession::progress() const {
  CalibrationProgress prog;
  if (samples_.empty()) return prog;

  float min_x = samples_.front().metrics.px;
  float min_y = samples_.front().metrics.py;
  float min_size = samples_.front().metrics.scale;
  float min_skew = samples_.front().metrics.skew;

  float max_x = min_x;
  float max_y = min_y;
  float max_size = min_size;
  float max_skew = min_skew;

  for (const auto& s : samples_) {
    min_x = std::min(min_x, s.metrics.px);
    min_y = std::min(min_y, s.metrics.py);
    min_size = std::min(min_size, s.metrics.scale);
    min_skew = std::min(min_skew, s.metrics.skew);

    max_x = std::max(max_x, s.metrics.px);
    max_y = std::max(max_y, s.metrics.py);
    max_size = std::max(max_size, s.metrics.scale);
    max_skew = std::max(max_skew, s.metrics.skew);
  }

  // Do not reward small size or skew
  min_size = 0.0f;
  min_skew = 0.0f;

  const float ranges[4] = {0.7f, 0.7f, 0.4f, 0.5f};
  prog.x = clamp01((max_x - min_x) / ranges[0]);
  prog.y = clamp01((max_y - min_y) / ranges[1]);
  prog.scale = clamp01((max_size - min_size) / ranges[2]);
  prog.skew = clamp01((max_skew - min_skew) / ranges[3]);
  return prog;
}

bool CalibrationSession::goodEnough() const {
  if (samples_.empty()) return false;
  if (samples_.size() >= 40) return true;
  CalibrationProgress prog = progress();
  return (prog.x >= 1.0f && prog.y >= 1.0f && prog.scale >= 1.0f && prog.skew >= 1.0f);
}

int CalibrationSession::sampleCount() const {
  return static_cast<int>(samples_.size());
}

const std::string& CalibrationSession::lastStatus() const {
  return last_status_;
}

const CalibrationResult& CalibrationSession::result() const {
  return result_;
}

bool CalibrationSession::calibrate(const cv::Size& image_size,
                                   const CalibrationProgressCallback& progress_callback) {
  auto fail = [&](std::string status, float progress) {
    last_status_ = std::move(status);
    reportCalibrationProgress(progress_callback, progress, last_status_);
    return false;
  };

  reportCalibrationProgress(progress_callback, 0.02f, "Validando muestras...");
  if (samples_.size() < 5) {
    return fail("Se necesitan al menos 5 muestras", 0.02f);
  }
  if (image_size.empty()) {
    return fail("Tamano de imagen invalido", 0.02f);
  }

  std::vector<PreparedMonoSample> prepared;
  prepared.reserve(samples_.size());
  std::vector<int> rejected_sample_indices;
  const size_t total_samples = samples_.size();
  const float prepare_begin = 0.08f;
  const float prepare_span = 0.18f;

  auto reportPreparationProgress = [&](size_t sample_index) {
    if (total_samples == 0U) return;
    const float fraction =
        static_cast<float>(sample_index + 1U) / static_cast<float>(total_samples);
    std::ostringstream message;
    message << "Preparando muestras " << (sample_index + 1U) << "/" << total_samples << "...";
    reportCalibrationProgress(progress_callback, prepare_begin + prepare_span * fraction, message.str());
  };

  if (config_.type == PatternType::Charuco) {
#ifdef HAVE_OPENCV_ARUCO
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(config_.aruco_dictionary);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::makePtr<cv::aruco::CharucoBoard>(
        config_.board_size, config_.square_size, config_.marker_size, dictionary);
    const auto board_corners = board->getChessboardCorners();

    for (size_t sample_idx = 0; sample_idx < samples_.size(); ++sample_idx) {
      const auto& s = samples_[sample_idx];
      const size_t n = std::min(s.corners.size(), s.ids.size());
      PreparedMonoSample prep;
      prep.sample_index = static_cast<int>(sample_idx);
      prep.object_points.reserve(n);
      prep.image_points.reserve(n);

      for (size_t i = 0; i < n; ++i) {
        const int id = s.ids[i];
        if (id < 0 || id >= static_cast<int>(board_corners.size())) continue;
        prep.object_points.push_back(board_corners[static_cast<size_t>(id)]);
        prep.image_points.push_back(s.corners[i]);
      }

      if (prep.object_points.size() < 4U) {
        rejected_sample_indices.push_back(static_cast<int>(sample_idx));
        reportPreparationProgress(sample_idx);
        continue;
      }
      prepared.push_back(std::move(prep));
      reportPreparationProgress(sample_idx);
    }
#else
    return fail("ChArUco requiere opencv_aruco", prepare_begin);
#endif
  } else {
    const std::vector<cv::Point3f> base_object_points = buildObjectPoints();
    if (base_object_points.empty()) {
      return fail("Patron invalido para calibracion", prepare_begin);
    }

    const size_t expected_points = base_object_points.size();
    for (size_t sample_idx = 0; sample_idx < samples_.size(); ++sample_idx) {
      const auto& s = samples_[sample_idx];
      if (s.corners.size() < expected_points) {
        rejected_sample_indices.push_back(static_cast<int>(sample_idx));
        reportPreparationProgress(sample_idx);
        continue;
      }
      PreparedMonoSample prep;
      prep.sample_index = static_cast<int>(sample_idx);
      prep.object_points = base_object_points;
      prep.image_points.assign(s.corners.begin(), s.corners.begin() + static_cast<long>(expected_points));
      prepared.push_back(std::move(prep));
      reportPreparationProgress(sample_idx);
    }
  }

  if (prepared.size() < 5U) {
    return fail("No hay suficientes muestras validas para calibrar", prepare_begin + prepare_span);
  }

  auto runSolve = [&](const std::vector<int>& active_indices,
                      bool use_warp_compensation) -> SolveOutput {
    SolveOutput out;
    if (active_indices.size() < 5U) {
      out.error = "Se necesitan al menos 5 muestras validas";
      return out;
    }

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    object_points.reserve(active_indices.size());
    image_points.reserve(active_indices.size());

    for (int idx : active_indices) {
      if (idx < 0 || idx >= static_cast<int>(prepared.size())) continue;
      object_points.push_back(prepared[static_cast<size_t>(idx)].object_points);
      image_points.push_back(prepared[static_cast<size_t>(idx)].image_points);
    }

    if (object_points.size() < 5U) {
      out.error = "Muestras insuficientes tras filtrar";
      return out;
    }

    cv::Mat k = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat d;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    try {
      if (camera_model_ == CameraModel::Fisheye) {
        d = cv::Mat::zeros(4, 1, CV_64F);
        int flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_FIX_SKEW;
        const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6);
        out.rms = cv::fisheye::calibrate(object_points, image_points, image_size,
                                         k, d, rvecs, tvecs, flags, criteria);
      } else {
        int flags = 0;
        if (pinhole_distortion_model_ == PinholeDistortionModel::RationalPolynomial) {
          flags |= cv::CALIB_RATIONAL_MODEL;
          d = cv::Mat::zeros(14, 1, CV_64F);
        } else {
          d = cv::Mat::zeros(8, 1, CV_64F);
        }

        if (use_warp_compensation &&
            config_.type != PatternType::Charuco &&
            hasSameObjectLayout(prepared, active_indices)) {
          const int fixed_point = std::clamp(config_.board_size.width - 1, 0,
                                             static_cast<int>(object_points.front().size()) - 1);
          cv::Mat refined_obj;
          const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6);
          out.rms = cv::calibrateCameraRO(object_points, image_points,
                                          image_size, fixed_point,
                                          k, d, rvecs, tvecs,
                                          refined_obj, flags, criteria);
          out.refined_object_points = refined_obj;
        } else {
          out.rms = cv::calibrateCamera(object_points, image_points, image_size,
                                        k, d, rvecs, tvecs, flags);
        }
      }
    } catch (const cv::Exception& e) {
      out.error = std::string("Fallo en calibracion: ") + e.what();
      return out;
    }

    out.ok = true;
    out.camera_matrix = k;
    out.dist_coeffs = d;
    out.rvecs = std::move(rvecs);
    out.tvecs = std::move(tvecs);
    return out;
  };

  std::vector<int> active_indices(prepared.size());
  std::iota(active_indices.begin(), active_indices.end(), 0);

  SolveOutput solve;
  double robust_threshold = 0.0;
  constexpr int kMaxRobustIterations = 6;
  const float robust_begin = 0.30f;
  const float robust_span = 0.42f;

  for (int iter = 0; iter < kMaxRobustIterations; ++iter) {
    const float iteration_base =
        robust_begin + robust_span * (static_cast<float>(iter) / static_cast<float>(kMaxRobustIterations));
    std::ostringstream solve_message;
    solve_message << "Resolviendo calibracion robusta (" << (iter + 1) << "/" << kMaxRobustIterations << ")...";
    reportCalibrationProgress(progress_callback, iteration_base, solve_message.str());

    solve = runSolve(active_indices, false);
    if (!solve.ok) {
      return fail(solve.error, iteration_base);
    }

    reportCalibrationProgress(progress_callback,
                              iteration_base + robust_span * (0.45f / static_cast<float>(kMaxRobustIterations)),
                              "Analizando errores por vista...");
    const std::vector<double> view_errors = computeViewErrors(
        prepared, active_indices,
        solve.rvecs, solve.tvecs,
        camera_model_, solve.camera_matrix, solve.dist_coeffs,
        nullptr, image_size);

    const double med = medianOf(view_errors);
    std::vector<double> abs_dev;
    abs_dev.reserve(view_errors.size());
    for (double e : view_errors) {
      abs_dev.push_back(std::abs(e - med));
    }
    const double mad = medianOf(abs_dev);
    const double sigma = std::max(1e-9, 1.4826 * mad);
    robust_threshold = med + std::max(0.10, 2.75 * sigma);

    std::vector<int> remove_positions;
    for (size_t i = 0; i < view_errors.size(); ++i) {
      if (view_errors[i] > robust_threshold && active_indices.size() - remove_positions.size() > 5U) {
        remove_positions.push_back(static_cast<int>(i));
      }
    }

    if (remove_positions.empty()) {
      reportCalibrationProgress(progress_callback,
                                iteration_base + robust_span *
                                                     (1.0f / static_cast<float>(kMaxRobustIterations)),
                                "Conjunto robusto estabilizado.");
      break;
    }

    std::sort(remove_positions.begin(), remove_positions.end(), std::greater<int>());
    for (int pos : remove_positions) {
      if (pos < 0 || pos >= static_cast<int>(active_indices.size())) continue;
      const int prepared_idx = active_indices[static_cast<size_t>(pos)];
      const int sample_idx = prepared[static_cast<size_t>(prepared_idx)].sample_index;
      rejected_sample_indices.push_back(sample_idx);
      active_indices.erase(active_indices.begin() + pos);
    }

    if (active_indices.size() < 5U) {
      return fail("Demasiados outliers: quedan menos de 5 muestras",
                  iteration_base + robust_span * (1.0f / static_cast<float>(kMaxRobustIterations)));
    }

    std::ostringstream reject_message;
    reject_message << "Descartando " << remove_positions.size() << " muestras atipicas...";
    reportCalibrationProgress(progress_callback,
                              iteration_base + robust_span * (1.0f / static_cast<float>(kMaxRobustIterations)),
                              reject_message.str());
  }

  // Final solve, enabling target-warp compensation if requested and applicable.
  reportCalibrationProgress(progress_callback,
                            0.78f,
                            target_warp_compensation_
                                ? "Resolviendo calibracion final con compensacion de warp..."
                                : "Resolviendo calibracion final...");
  solve = runSolve(active_indices, target_warp_compensation_);
  if (!solve.ok) {
    return fail(solve.error, 0.78f);
  }

  ResidualGrid grid;
  grid.cols = 16;
  grid.rows = 12;
  reportCalibrationProgress(progress_callback, 0.88f, "Calculando errores de reproyeccion...");
  std::vector<double> final_view_errors = computeViewErrors(
      prepared, active_indices,
      solve.rvecs, solve.tvecs,
      camera_model_, solve.camera_matrix, solve.dist_coeffs,
      &grid, image_size);

  std::vector<int> inlier_sample_indices;
  inlier_sample_indices.reserve(active_indices.size());
  for (int prepared_idx : active_indices) {
    inlier_sample_indices.push_back(prepared[static_cast<size_t>(prepared_idx)].sample_index);
  }

  std::sort(rejected_sample_indices.begin(), rejected_sample_indices.end());
  rejected_sample_indices.erase(std::unique(rejected_sample_indices.begin(), rejected_sample_indices.end()),
                                rejected_sample_indices.end());

  reportCalibrationProgress(progress_callback, 0.95f, "Consolidando resultado...");
  result_ = CalibrationResult{};
  result_.valid = true;
  result_.rms = solve.rms;
  result_.model = camera_model_;
  result_.pinhole_distortion = pinhole_distortion_model_;
  result_.camera_matrix = solve.camera_matrix;
  result_.dist_coeffs = solve.dist_coeffs;
  result_.refined_object_points = solve.refined_object_points;
  result_.image_size = image_size;
  result_.target_warp_compensation_used =
      (camera_model_ == CameraModel::Pinhole && target_warp_compensation_ &&
       config_.type != PatternType::Charuco && !solve.refined_object_points.empty());
  result_.per_view_errors = std::move(final_view_errors);
  result_.inlier_sample_indices = std::move(inlier_sample_indices);
  result_.rejected_sample_indices = std::move(rejected_sample_indices);
  result_.mean_reprojection_error = meanOf(result_.per_view_errors);
  result_.median_reprojection_error = medianOf(result_.per_view_errors);
  result_.p95_reprojection_error = percentileOf(result_.per_view_errors, 0.95);
  result_.outlier_threshold = robust_threshold;
  result_.residual_grid = std::move(grid);

  std::ostringstream status;
  status << "Calibracion completada: "
         << "inliers=" << result_.inlier_sample_indices.size()
         << ", rechazadas=" << result_.rejected_sample_indices.size()
         << ", RMS=" << result_.rms;
  if (camera_model_ == CameraModel::Fisheye) {
    status << " (fisheye)";
  } else {
    status << " (" << pinholeModelName(pinhole_distortion_model_) << ")";
  }
  if (result_.target_warp_compensation_used) {
    status << ", warp-comp=on";
  }
  last_status_ = status.str();
  reportCalibrationProgress(progress_callback, 1.0f, "Calibracion completada.");
  return true;
}

bool CalibrationSession::saveResult(const std::string& path) const {
  if (!result_.valid) return false;
  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) return false;

  fs << "image_width" << result_.image_size.width;
  fs << "image_height" << result_.image_size.height;
  fs << "camera_model" << cameraModelName(result_.model);

  std::string distortion_model;
  if (result_.model == CameraModel::Fisheye) {
    distortion_model = "fisheye";
  } else {
    distortion_model = pinholeModelName(result_.pinhole_distortion);
  }
  fs << "distortion_model" << distortion_model;
  fs << "camera_matrix" << result_.camera_matrix;
  fs << "distortion_coefficients" << result_.dist_coeffs;
  fs << "rms" << result_.rms;

  fs << "target_warp_compensation_used" << result_.target_warp_compensation_used;
  if (!result_.refined_object_points.empty()) {
    fs << "refined_object_points" << result_.refined_object_points;
  }

  fs << "mean_reprojection_error" << result_.mean_reprojection_error;
  fs << "median_reprojection_error" << result_.median_reprojection_error;
  fs << "p95_reprojection_error" << result_.p95_reprojection_error;
  fs << "outlier_threshold" << result_.outlier_threshold;

  appendIntSequence(fs, "inlier_sample_indices", result_.inlier_sample_indices);
  appendIntSequence(fs, "rejected_sample_indices", result_.rejected_sample_indices);
  appendDoubleSequence(fs, "view_errors", result_.per_view_errors);

  fs << "residual_grid_cols" << result_.residual_grid.cols;
  fs << "residual_grid_rows" << result_.residual_grid.rows;
  appendFloatSequence(fs, "residual_grid_mean_error", result_.residual_grid.mean_error);
  appendFloatSequence(fs, "residual_grid_mean_dx", result_.residual_grid.mean_dx);
  appendFloatSequence(fs, "residual_grid_mean_dy", result_.residual_grid.mean_dy);

  fs.release();
  return true;
}

SampleMetrics CalibrationSession::computeMetrics(const std::vector<cv::Point2f>& corners,
                                                 const std::vector<int>& ids,
                                                 const cv::Size& image_size) const {
  SampleMetrics metrics;
  if (corners.empty() || image_size.area() <= 0) return metrics;

  std::array<cv::Point2f, 4> outside{};
  if (!getOutsideCorners(corners, ids, outside)) return metrics;

  double sum_x = 0.0;
  double sum_y = 0.0;
  for (const auto& p : corners) {
    sum_x += p.x;
    sum_y += p.y;
  }
  double mean_x = sum_x / corners.size();
  double mean_y = sum_y / corners.size();

  float area = calculate_area(outside);
  float skew = calculate_skew(outside);

  double border = std::sqrt(area);
  double width = static_cast<double>(image_size.width);
  double height = static_cast<double>(image_size.height);

  metrics.px = clamp01(static_cast<float>((mean_x - border / 2.0) / (width - border)));
  metrics.py = clamp01(static_cast<float>((mean_y - border / 2.0) / (height - border)));
  metrics.scale = clamp01(static_cast<float>(std::sqrt(area / (width * height))));
  metrics.skew = skew;

  return metrics;
}

std::vector<cv::Point3f> CalibrationSession::buildObjectPoints() const {
  std::vector<cv::Point3f> obj;
  obj.reserve(config_.board_size.area());
  for (int i = 0; i < config_.board_size.height; ++i) {
    for (int j = 0; j < config_.board_size.width; ++j) {
      float x = 0.0f;
      float y = 0.0f;
      if (config_.type == PatternType::CirclesAsymmetric) {
        x = (2.0f * j + (i % 2)) * config_.square_size;
        y = i * config_.square_size;
      } else {
        x = j * config_.square_size;
        y = i * config_.square_size;
      }
      obj.emplace_back(x, y, 0.0f);
    }
  }
  return obj;
}

bool CalibrationSession::isDistinct(const SampleMetrics& m) const {
  float best = std::numeric_limits<float>::infinity();
  for (const auto& s : samples_) {
    float dist = std::abs(m.px - s.metrics.px) +
                 std::abs(m.py - s.metrics.py) +
                 std::abs(m.scale - s.metrics.scale) +
                 std::abs(m.skew - s.metrics.skew);
    best = std::min(best, dist);
  }
  return best > 0.2f;
}

bool CalibrationSession::getOutsideCorners(const std::vector<cv::Point2f>& corners,
                                           const std::vector<int>& ids,
                                           std::array<cv::Point2f, 4>& out) const {
  if (config_.type == PatternType::Charuco) {
    if (ids.empty()) return false;
    int xdim = config_.board_size.width - 1;
    int ydim = config_.board_size.height - 1;
    if (xdim <= 0 || ydim <= 0) return false;

    std::vector<std::vector<bool>> board_vis(ydim, std::vector<bool>(xdim, false));
    for (int id : ids) {
      if (id >= 0 && id < xdim * ydim) {
        int y = id / xdim;
        int x = id % xdim;
        board_vis[y][x] = true;
      }
    }

    int best_area = 0;
    int best_x1 = -1;
    int best_x2 = -1;
    int best_y1 = -1;
    int best_y2 = -1;
    for (int x1 = 0; x1 < xdim; ++x1) {
      for (int x2 = x1; x2 < xdim; ++x2) {
        for (int y1 = 0; y1 < ydim; ++y1) {
          for (int y2 = y1; y2 < ydim; ++y2) {
            if (board_vis[y1][x1] && board_vis[y1][x2] &&
                board_vis[y2][x1] && board_vis[y2][x2]) {
              int area = (x2 - x1 + 1) * (y2 - y1 + 1);
              if (area > best_area) {
                best_area = area;
                best_x1 = x1;
                best_x2 = x2;
                best_y1 = y1;
                best_y2 = y2;
              }
            }
          }
        }
      }
    }

    if (best_area == 0) return false;

    int corner_ids[4] = {
        best_y2 * xdim + best_x1,
        best_y2 * xdim + best_x2,
        best_y1 * xdim + best_x2,
        best_y1 * xdim + best_x1,
    };

    for (int i = 0; i < 4; ++i) {
      auto it = std::find(ids.begin(), ids.end(), corner_ids[i]);
      if (it == ids.end()) return false;
      size_t index = static_cast<size_t>(std::distance(ids.begin(), it));
      if (index >= corners.size()) return false;
      out[i] = corners[index];
    }
    return true;
  }

  int cols = config_.board_size.width;
  int rows = config_.board_size.height;
  if (cols <= 1 || rows <= 1) return false;
  if (corners.size() < static_cast<size_t>(cols * rows)) return false;

  out[0] = corners[0];
  out[1] = corners[cols - 1];
  out[2] = corners[cols * rows - 1];
  out[3] = corners[(rows - 1) * cols];
  return true;
}

void StereoCalibrationSession::setPattern(const PatternConfig& config) {
  if (config_.type != config.type || config_.board_size != config.board_size ||
      std::abs(config_.square_size - config.square_size) > 1e-6f ||
      std::abs(config_.marker_size - config.marker_size) > 1e-6f ||
      config_.aruco_dictionary != config.aruco_dictionary) {
    config_ = config;
    clear();
  }
}

void StereoCalibrationSession::clear() {
  samples_.clear();
  signatures_.clear();
  result_ = StereoCalibrationResult{};
  last_status_.clear();
}

void StereoCalibrationSession::setExpectedBaseline(double baseline_m) {
  expected_baseline_m_ = std::max(0.0, baseline_m);
}

bool StereoCalibrationSession::addSample(const std::vector<cv::Point2f>& left_corners,
                                         const std::vector<int>& left_ids,
                                         const cv::Size& left_image_size,
                                         const std::vector<cv::Point2f>& right_corners,
                                         const std::vector<int>& right_ids,
                                         const cv::Size& right_image_size,
                                         bool allow_duplicates) {
  if (left_image_size.empty() || right_image_size.empty()) {
    last_status_ = "Imagen invalida para par estereo";
    return false;
  }
  if (left_image_size != right_image_size) {
    last_status_ = "Resoluciones distintas entre camaras";
    return false;
  }

  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> left_points;
  std::vector<cv::Point2f> right_points;
  if (!buildStereoPoints(left_corners, left_ids, right_corners, right_ids,
                         object_points, left_points, right_points)) {
    return false;
  }

  SampleMetrics left_metrics = computeSimpleMetrics(left_points, left_image_size);
  SampleMetrics right_metrics = computeSimpleMetrics(right_points, right_image_size);
  StereoSignature signature = buildSignature(left_metrics, right_metrics);
  if (!allow_duplicates && !isDistinct(signature)) {
    last_status_ = "Muestra estereo similar, no agregada";
    return false;
  }

  StereoSample sample;
  sample.object_points = std::move(object_points);
  sample.left_points = std::move(left_points);
  sample.right_points = std::move(right_points);
  sample.image_size = left_image_size;
  sample.left_metrics = left_metrics;
  sample.right_metrics = right_metrics;
  samples_.push_back(std::move(sample));
  signatures_.push_back(signature);
  last_status_ = "Muestra estereo agregada";
  return true;
}

int StereoCalibrationSession::sampleCount() const {
  return static_cast<int>(samples_.size());
}

const std::string& StereoCalibrationSession::lastStatus() const {
  return last_status_;
}

const StereoCalibrationResult& StereoCalibrationSession::result() const {
  return result_;
}

bool StereoCalibrationSession::calibrate(bool fix_intrinsics) {
  if (samples_.size() < 5) {
    last_status_ = "Se necesitan al menos 5 muestras estereo";
    return false;
  }

  std::vector<std::vector<cv::Point3f>> object_points;
  std::vector<std::vector<cv::Point2f>> image_points_left;
  std::vector<std::vector<cv::Point2f>> image_points_right;
  object_points.reserve(samples_.size());
  image_points_left.reserve(samples_.size());
  image_points_right.reserve(samples_.size());

  const cv::Size image_size = samples_.front().image_size;
  for (const auto& sample : samples_) {
    if (sample.image_size != image_size) {
      last_status_ = "Las muestras estereo deben tener la misma resolucion";
      return false;
    }
    object_points.push_back(sample.object_points);
    image_points_left.push_back(sample.left_points);
    image_points_right.push_back(sample.right_points);
  }

  cv::Mat k_left = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat d_left = cv::Mat::zeros(8, 1, CV_64F);
  cv::Mat k_right = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat d_right = cv::Mat::zeros(8, 1, CV_64F);
  cv::Mat r;
  cv::Mat t;
  cv::Mat e;
  cv::Mat f;

  int flags = 0;

  try {
    if (fix_intrinsics) {
      std::vector<cv::Mat> rvecs;
      std::vector<cv::Mat> tvecs;
      cv::calibrateCamera(object_points, image_points_left, image_size, k_left, d_left, rvecs, tvecs);
      cv::calibrateCamera(object_points, image_points_right, image_size, k_right, d_right, rvecs, tvecs);
      flags |= cv::CALIB_FIX_INTRINSIC;
    }

    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6);
    double rms = cv::stereoCalibrate(object_points, image_points_left, image_points_right,
                                     k_left, d_left, k_right, d_right,
                                     image_size, r, t, e, f, flags, criteria);

    cv::Mat r1;
    cv::Mat r2;
    cv::Mat p1;
    cv::Mat p2;
    cv::Mat q;
    cv::Rect roi_left;
    cv::Rect roi_right;
    cv::stereoRectify(k_left, d_left, k_right, d_right,
                      image_size, r, t,
                      r1, r2, p1, p2, q,
                      cv::CALIB_ZERO_DISPARITY,
                      0.0,
                      image_size,
                      &roi_left,
                      &roi_right);

    cv::Mat map1_left;
    cv::Mat map2_left;
    cv::Mat map1_right;
    cv::Mat map2_right;
    cv::initUndistortRectifyMap(k_left, d_left, r1, p1, image_size, CV_32FC1, map1_left, map2_left);
    cv::initUndistortRectifyMap(k_right, d_right, r2, p2, image_size, CV_32FC1, map1_right, map2_right);

    result_ = StereoCalibrationResult{};
    result_.valid = true;
    result_.rms = rms;
    result_.camera_matrix_left = k_left;
    result_.dist_coeffs_left = d_left;
    result_.camera_matrix_right = k_right;
    result_.dist_coeffs_right = d_right;
    result_.rotation = r;
    result_.translation = t;
    result_.essential = e;
    result_.fundamental = f;
    result_.rectification_left = r1;
    result_.rectification_right = r2;
    result_.projection_left = p1;
    result_.projection_right = p2;
    result_.disparity_to_depth = q;
    result_.rect_map1_left = map1_left;
    result_.rect_map2_left = map2_left;
    result_.rect_map1_right = map1_right;
    result_.rect_map2_right = map2_right;
    result_.valid_roi_left = roi_left;
    result_.valid_roi_right = roi_right;
    result_.image_size = image_size;
    result_.baseline = cv::norm(t);

  } catch (const cv::Exception& e) {
    last_status_ = std::string("Fallo en calibracion estereo: ") + e.what();
    return false;
  }

  std::ostringstream status;
  status << "Calibracion estereo completada (RMS=" << result_.rms << ")";
  if (expected_baseline_m_ > 0.0) {
    const double err_m = std::abs(result_.baseline - expected_baseline_m_);
    const double err_pct = (err_m / expected_baseline_m_) * 100.0;
    status << ", baseline=" << result_.baseline << " m, error=" << err_pct << "%";
  }
  last_status_ = status.str();
  return true;
}

bool StereoCalibrationSession::saveResult(const std::string& path) const {
  if (!result_.valid) return false;

  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) return false;

  fs << "image_width" << result_.image_size.width;
  fs << "image_height" << result_.image_size.height;
  fs << "camera_matrix_left" << result_.camera_matrix_left;
  fs << "distortion_coefficients_left" << result_.dist_coeffs_left;
  fs << "camera_matrix_right" << result_.camera_matrix_right;
  fs << "distortion_coefficients_right" << result_.dist_coeffs_right;
  fs << "rotation_matrix" << result_.rotation;
  fs << "translation_vector" << result_.translation;
  fs << "essential_matrix" << result_.essential;
  fs << "fundamental_matrix" << result_.fundamental;
  fs << "rectification_left" << result_.rectification_left;
  fs << "rectification_right" << result_.rectification_right;
  fs << "projection_left" << result_.projection_left;
  fs << "projection_right" << result_.projection_right;
  fs << "disparity_to_depth" << result_.disparity_to_depth;
  fs << "rectification_map1_left" << result_.rect_map1_left;
  fs << "rectification_map2_left" << result_.rect_map2_left;
  fs << "rectification_map1_right" << result_.rect_map1_right;
  fs << "rectification_map2_right" << result_.rect_map2_right;

  fs << "valid_roi_left_x" << result_.valid_roi_left.x;
  fs << "valid_roi_left_y" << result_.valid_roi_left.y;
  fs << "valid_roi_left_width" << result_.valid_roi_left.width;
  fs << "valid_roi_left_height" << result_.valid_roi_left.height;
  fs << "valid_roi_right_x" << result_.valid_roi_right.x;
  fs << "valid_roi_right_y" << result_.valid_roi_right.y;
  fs << "valid_roi_right_width" << result_.valid_roi_right.width;
  fs << "valid_roi_right_height" << result_.valid_roi_right.height;

  fs << "rms" << result_.rms;
  fs << "baseline_m" << result_.baseline;
  fs << "expected_baseline_m" << expected_baseline_m_;
  fs.release();
  return true;
}

SampleMetrics StereoCalibrationSession::computeSimpleMetrics(const std::vector<cv::Point2f>& corners,
                                                             const cv::Size& image_size) const {
  SampleMetrics out;
  if (corners.empty() || image_size.empty()) return out;

  float min_x = corners.front().x;
  float min_y = corners.front().y;
  float max_x = min_x;
  float max_y = min_y;
  double mean_x = 0.0;
  double mean_y = 0.0;

  for (const auto& p : corners) {
    min_x = std::min(min_x, p.x);
    min_y = std::min(min_y, p.y);
    max_x = std::max(max_x, p.x);
    max_y = std::max(max_y, p.y);
    mean_x += p.x;
    mean_y += p.y;
  }

  mean_x /= static_cast<double>(corners.size());
  mean_y /= static_cast<double>(corners.size());
  const float area = std::max(0.0f, (max_x - min_x) * (max_y - min_y));
  const float image_area = static_cast<float>(image_size.area());

  out.px = clamp01(static_cast<float>(mean_x / image_size.width));
  out.py = clamp01(static_cast<float>(mean_y / image_size.height));
  out.scale = clamp01(image_area > 0.0f ? std::sqrt(area / image_area) : 0.0f);
  out.skew = 0.0f;
  return out;
}

StereoCalibrationSession::StereoSignature StereoCalibrationSession::buildSignature(
    const SampleMetrics& left, const SampleMetrics& right) const {
  StereoSignature sig;
  sig.left_px = left.px;
  sig.left_py = left.py;
  sig.left_scale = left.scale;
  sig.right_px = right.px;
  sig.right_py = right.py;
  sig.right_scale = right.scale;
  return sig;
}

bool StereoCalibrationSession::isDistinct(const StereoSignature& signature) const {
  float best = std::numeric_limits<float>::infinity();
  for (const auto& s : signatures_) {
    const float dist = std::abs(signature.left_px - s.left_px) +
                       std::abs(signature.left_py - s.left_py) +
                       std::abs(signature.left_scale - s.left_scale) +
                       std::abs(signature.right_px - s.right_px) +
                       std::abs(signature.right_py - s.right_py) +
                       std::abs(signature.right_scale - s.right_scale);
    best = std::min(best, dist);
  }
  return best > 0.3f;
}

bool StereoCalibrationSession::buildRegularObjectPoints(std::vector<cv::Point3f>& out) const {
  if (config_.board_size.width <= 1 || config_.board_size.height <= 1) {
    return false;
  }

  out.clear();
  out.reserve(static_cast<size_t>(config_.board_size.area()));
  for (int i = 0; i < config_.board_size.height; ++i) {
    for (int j = 0; j < config_.board_size.width; ++j) {
      float x = 0.0f;
      float y = 0.0f;
      if (config_.type == PatternType::CirclesAsymmetric) {
        x = (2.0f * j + (i % 2)) * config_.square_size;
        y = i * config_.square_size;
      } else {
        x = j * config_.square_size;
        y = i * config_.square_size;
      }
      out.emplace_back(x, y, 0.0f);
    }
  }
  return true;
}

bool StereoCalibrationSession::buildStereoPoints(const std::vector<cv::Point2f>& left_corners,
                                                 const std::vector<int>& left_ids,
                                                 const std::vector<cv::Point2f>& right_corners,
                                                 const std::vector<int>& right_ids,
                                                 std::vector<cv::Point3f>& object_points,
                                                 std::vector<cv::Point2f>& left_points,
                                                 std::vector<cv::Point2f>& right_points) {
  if (config_.type == PatternType::Charuco) {
#ifdef HAVE_OPENCV_ARUCO
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(config_.aruco_dictionary);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::makePtr<cv::aruco::CharucoBoard>(
        config_.board_size, config_.square_size, config_.marker_size, dictionary);
    const auto board_corners = board->getChessboardCorners();

    std::unordered_map<int, cv::Point2f> left_map;
    std::unordered_map<int, cv::Point2f> right_map;

    const size_t left_n = std::min(left_corners.size(), left_ids.size());
    for (size_t i = 0; i < left_n; ++i) {
      left_map[left_ids[i]] = left_corners[i];
    }

    const size_t right_n = std::min(right_corners.size(), right_ids.size());
    for (size_t i = 0; i < right_n; ++i) {
      right_map[right_ids[i]] = right_corners[i];
    }

    std::vector<int> common_ids;
    common_ids.reserve(std::min(left_map.size(), right_map.size()));
    for (const auto& kv : left_map) {
      if (right_map.find(kv.first) != right_map.end()) {
        if (kv.first >= 0 && kv.first < static_cast<int>(board_corners.size())) {
          common_ids.push_back(kv.first);
        }
      }
    }
    std::sort(common_ids.begin(), common_ids.end());

    if (common_ids.size() < 6U) {
      last_status_ = "ChArUco estereo requiere al menos 6 esquinas comunes";
      return false;
    }

    object_points.clear();
    left_points.clear();
    right_points.clear();
    object_points.reserve(common_ids.size());
    left_points.reserve(common_ids.size());
    right_points.reserve(common_ids.size());

    for (int id : common_ids) {
      object_points.push_back(board_corners[static_cast<size_t>(id)]);
      left_points.push_back(left_map[id]);
      right_points.push_back(right_map[id]);
    }
    return true;
#else
    last_status_ = "Modo estereo con ChArUco requiere opencv_aruco";
    return false;
#endif
  }

  if (!buildRegularObjectPoints(object_points)) {
    last_status_ = "Configuracion de patron invalida";
    return false;
  }

  const size_t n = object_points.size();
  if (left_corners.size() < n || right_corners.size() < n) {
    last_status_ = "No hay suficientes puntos detectados en ambos frames";
    return false;
  }

  left_points.assign(left_corners.begin(), left_corners.begin() + static_cast<long>(n));
  right_points.assign(right_corners.begin(), right_corners.begin() + static_cast<long>(n));
  return true;
}
