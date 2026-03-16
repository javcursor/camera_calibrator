#pragma once

#include <opencv2/core.hpp>

enum class PatternType {
  ChessboardAuto,
  Chessboard,
  ChessboardSB,
  CirclesSymmetric,
  CirclesAsymmetric,
  Charuco,
};

struct PatternConfig {
  PatternType type = PatternType::ChessboardAuto;
  cv::Size board_size = {9, 6};
  float square_size = 0.025f;
  float marker_size = 0.0125f;  // Charuco only
  int aruco_dictionary = 0;     // Charuco only (OpenCV predefined dict id)
};
