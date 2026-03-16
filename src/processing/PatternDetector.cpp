#include "processing/PatternDetector.hpp"

#ifdef HAVE_OPENCV_ARUCO
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <limits>

PatternDetector::PatternDetector(const PatternConfig& config) : config_(config) {}

void PatternDetector::setConfig(const PatternConfig& config) {
  config_ = config;
}

const PatternConfig& PatternDetector::config() const {
  return config_;
}

namespace {
bool detectChessboardClassic(const cv::Mat& gray, const cv::Size& board_size,
                             std::vector<cv::Point2f>& corners) {
  const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
  const bool found = cv::findChessboardCorners(gray, board_size, corners, flags);
  if (found) {
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
  }
  return found;
}

bool detectChessboardSB(const cv::Mat& gray, const cv::Size& board_size,
                        std::vector<cv::Point2f>& corners) {
#if (CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 5)
  return cv::findChessboardCornersSB(gray, board_size, corners, cv::CALIB_CB_NORMALIZE_IMAGE);
#else
  return detectChessboardClassic(gray, board_size, corners);
#endif
}

cv::Ptr<cv::FeatureDetector> makeCirclesBlobDetector(const cv::Mat& gray) {
  cv::SimpleBlobDetector::Params params;
  params.minThreshold = 5.0f;
  params.maxThreshold = 220.0f;
  params.thresholdStep = 5.0f;
  params.minRepeatability = 1;
  params.minDistBetweenBlobs = 4.0f;
  params.filterByColor = true;
  params.blobColor = 0;
  params.filterByArea = true;
  params.minArea = 8.0f;
  params.maxArea = static_cast<float>(std::max<int64_t>(2000, gray.total() / 2));
  params.filterByCircularity = false;
  params.filterByInertia = false;
  params.filterByConvexity = false;
  return cv::SimpleBlobDetector::create(params);
}

std::vector<cv::Point2f> extractCircleCandidatesFromBinary(const cv::Mat& binary) {
  std::vector<cv::Point2f> centers;
  if (binary.empty()) return centers;

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  const double image_area = static_cast<double>(binary.rows) * static_cast<double>(binary.cols);
  const double min_area = std::max(10.0, image_area * 0.00002);
  const double max_area = image_area * 0.10;

  for (const auto& contour : contours) {
    const double area = std::abs(cv::contourArea(contour));
    if (area < min_area || area > max_area) continue;

    const double perimeter = cv::arcLength(contour, true);
    if (perimeter <= 1e-6) continue;

    const double circularity = (4.0 * CV_PI * area) / (perimeter * perimeter);
    if (circularity < 0.45) continue;

    const cv::Rect bounds = cv::boundingRect(contour);
    if (bounds.width < 3 || bounds.height < 3) continue;
    const double aspect = static_cast<double>(bounds.width) / static_cast<double>(bounds.height);
    if (aspect < 0.5 || aspect > 1.5) continue;

    const cv::Moments moments = cv::moments(contour);
    if (std::abs(moments.m00) <= std::numeric_limits<double>::epsilon()) continue;

    const cv::Point2f center(static_cast<float>(moments.m10 / moments.m00),
                             static_cast<float>(moments.m01 / moments.m00));

    bool duplicate = false;
    for (const auto& existing : centers) {
      if (cv::norm(existing - center) < 4.0f) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      centers.push_back(center);
    }
  }

  return centers;
}

std::vector<std::vector<cv::Point2f>> buildCircleCandidateSets(const cv::Mat& gray) {
  std::vector<std::vector<cv::Point2f>> candidate_sets;
  if (gray.empty()) return candidate_sets;

  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0.0);

  std::vector<cv::Mat> binaries;

  cv::Mat otsu_inv;
  cv::threshold(blurred, otsu_inv, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  binaries.push_back(otsu_inv);

  cv::Mat adaptive_inv;
  cv::adaptiveThreshold(blurred, adaptive_inv, 255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 31, 7.0);
  binaries.push_back(adaptive_inv);

  cv::Mat equalized;
  cv::equalizeHist(gray, equalized);
  cv::GaussianBlur(equalized, equalized, cv::Size(5, 5), 0.0);

  cv::Mat equalized_otsu_inv;
  cv::threshold(equalized, equalized_otsu_inv, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  binaries.push_back(equalized_otsu_inv);

  cv::Mat equalized_adaptive_inv;
  cv::adaptiveThreshold(equalized, equalized_adaptive_inv, 255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 31, 5.0);
  binaries.push_back(equalized_adaptive_inv);

  for (const auto& binary : binaries) {
    std::vector<cv::Point2f> candidates = extractCircleCandidatesFromBinary(binary);
    if (!candidates.empty()) {
      candidate_sets.push_back(std::move(candidates));
    }
  }

  return candidate_sets;
}

bool detectCirclesGridFromCandidates(const std::vector<cv::Point2f>& candidates,
                                     const cv::Size& board_size,
                                     bool asymmetric,
                                     std::vector<cv::Point2f>& corners) {
  if (candidates.size() < static_cast<size_t>(board_size.area())) {
    return false;
  }

  const int base_flags = asymmetric ? cv::CALIB_CB_ASYMMETRIC_GRID : cv::CALIB_CB_SYMMETRIC_GRID;
  cv::CirclesGridFinderParameters parameters;
  parameters.gridType = asymmetric
                            ? cv::CirclesGridFinderParameters::ASYMMETRIC_GRID
                            : cv::CirclesGridFinderParameters::SYMMETRIC_GRID;

  corners.clear();
  if (cv::findCirclesGrid(candidates, board_size, corners, base_flags,
                          cv::Ptr<cv::FeatureDetector>(), parameters)) {
    return true;
  }

  corners.clear();
  return cv::findCirclesGrid(candidates, board_size, corners, base_flags | cv::CALIB_CB_CLUSTERING,
                             cv::Ptr<cv::FeatureDetector>(), parameters);
}

bool detectCirclesGridRobust(const cv::Mat& gray,
                             const cv::Size& board_size,
                             bool asymmetric,
                             std::vector<cv::Point2f>& corners) {
  if (gray.empty()) return false;

  const int base_flags = asymmetric ? cv::CALIB_CB_ASYMMETRIC_GRID : cv::CALIB_CB_SYMMETRIC_GRID;
  const auto tryDetect = [&](const cv::Mat& input, int flags) {
    corners.clear();
    cv::Ptr<cv::FeatureDetector> blob_detector = makeCirclesBlobDetector(input);
    return cv::findCirclesGrid(input, board_size, corners, flags, blob_detector);
  };

  if (tryDetect(gray, base_flags | cv::CALIB_CB_CLUSTERING)) {
    return true;
  }
  if (tryDetect(gray, base_flags)) {
    return true;
  }

  cv::Mat equalized;
  cv::equalizeHist(gray, equalized);
  if (tryDetect(equalized, base_flags | cv::CALIB_CB_CLUSTERING)) {
    return true;
  }
  if (tryDetect(equalized, base_flags)) {
    return true;
  }

  const auto candidate_sets = buildCircleCandidateSets(gray);
  for (const auto& candidates : candidate_sets) {
    if (detectCirclesGridFromCandidates(candidates, board_size, asymmetric, corners)) {
      return true;
    }
  }

  return false;
}
}  // namespace

DetectionResult PatternDetector::detect(const cv::Mat& gray) const {
  DetectionResult result;
  if (gray.empty()) return result;

  switch (config_.type) {
    case PatternType::ChessboardAuto: {
      result.found = detectChessboardSB(gray, config_.board_size, result.corners);
      if (!result.found) {
        result.corners.clear();
        result.found = detectChessboardClassic(gray, config_.board_size, result.corners);
      }
      break;
    }
    case PatternType::Chessboard: {
      result.found = detectChessboardClassic(gray, config_.board_size, result.corners);
      break;
    }
    case PatternType::ChessboardSB: {
      result.found = detectChessboardSB(gray, config_.board_size, result.corners);
      break;
    }
    case PatternType::CirclesSymmetric:
    case PatternType::CirclesAsymmetric: {
      result.found = detectCirclesGridRobust(
          gray, config_.board_size, config_.type == PatternType::CirclesAsymmetric, result.corners);
      break;
    }
    case PatternType::Charuco: {
#ifdef HAVE_OPENCV_ARUCO
      cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(config_.aruco_dictionary);
      cv::Ptr<cv::aruco::Dictionary> dictionary_ptr = cv::makePtr<cv::aruco::Dictionary>(dictionary);
      cv::Ptr<cv::aruco::CharucoBoard> board = cv::makePtr<cv::aruco::CharucoBoard>(
          config_.board_size, config_.square_size, config_.marker_size, dictionary);

      std::vector<std::vector<cv::Point2f>> marker_corners;
      std::vector<int> marker_ids;
      cv::aruco::detectMarkers(gray, dictionary_ptr, marker_corners, marker_ids);
      if (marker_ids.empty()) break;

      cv::Mat charuco_corners;
      cv::Mat charuco_ids;
      cv::aruco::interpolateCornersCharuco(marker_corners, marker_ids, gray, board,
                                           charuco_corners, charuco_ids);
      if (charuco_ids.total() < 4) break;

      result.corners.reserve(charuco_corners.total());
      result.ids.reserve(charuco_ids.total());
      for (int i = 0; i < charuco_ids.rows; ++i) {
        result.ids.push_back(charuco_ids.at<int>(i));
        result.corners.push_back(charuco_corners.at<cv::Point2f>(i));
      }
      result.found = true;
#else
      result.found = false;
#endif
      break;
    }
  }

  return result;
}
