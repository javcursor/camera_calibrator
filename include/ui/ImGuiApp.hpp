#pragma once

#include "camera/CameraDevice.hpp"
#include "processing/Calibration.hpp"
#include "processing/PatternDetector.hpp"
#include "ui/ImageTexture.hpp"

#include <opencv2/opencv.hpp>

#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class ImGuiApp {
 public:
  enum class OfflineImageEncoding {
    Standard = 0,
    BayerRG = 1,
    BayerBG = 2,
    BayerGB = 3,
    BayerGR = 4,
  };

  int run();

 private:
  enum class UiTheme {
    Dark = 0,
    Light = 1,
  };

  enum class CalibrationInputMode {
    Live = 0,
    ImageFolder = 1,
    VideoFile = 2,
  };

  enum class OfflinePathPickerTarget {
    None = 0,
    ImageFolder = 1,
    VideoFile = 2,
  };

  enum class OfflineDetectionState {
    Idle = 0,
    Detecting = 1,
    Ready = 2,
  };

  struct FeatureTask {
    enum class Type {
      Refresh,
      SetValue,
      ExecuteCommand,
    };

    Type type = Type::Refresh;
    FeatureInfo feature;
    FeatureValue value;
  };

  struct CameraSlot {
    CameraDevice device;
    std::vector<CameraInfo> devices;
    int selected_device = -1;
    std::vector<FeatureInfo> features;
    std::unordered_map<std::string, FeatureValue> feature_values;
    std::string feature_filter;
    std::string feature_status;
    bool feature_busy = false;
    bool feature_refresh_queued = false;
    std::deque<FeatureTask> feature_tasks;
    std::mutex feature_mutex;
    std::condition_variable feature_cv;
    std::thread feature_worker;
    bool feature_worker_stop = false;
    bool feature_worker_running = false;
    std::mutex frame_mutex;
    std::thread processing_worker;
    std::atomic<bool> processing_worker_stop{false};
    bool processing_worker_running = false;
    ImageTexture texture;
    cv::Mat display_frame;
    cv::Mat last_frame_raw;
    cv::Size last_image_size;
    DetectionResult detection;
    uint64_t detection_seq = 0;
    uint64_t detection_timestamp_ns = 0;
    SampleMetrics metrics{};
    bool found = false;
    double processing_fps = 0.0;
    uint64_t processed_frames = 0;
    std::atomic<bool> bayer_swap_rb{true};
    bool ir_mode = false;
    bool invert_ir = false;
    bool use_gstreamer_pipeline = false;
    std::string gstreamer_pipeline;
    std::string gentl_cti_path;
    bool open_config_dialog = false;
  };

  struct OfflineImageLoadTask {
    std::mutex mutex;
    std::thread worker;
    bool running = false;
    bool result_ready = false;
    bool show_popup = false;
    std::string target_dir;
    std::string progress_message;
    size_t scanned_entries = 0;
    size_t found_images = 0;
    std::vector<std::string> files;
    std::string result_status;
  };

  struct OfflineVideoOpenTask {
    std::mutex mutex;
    std::thread worker;
    bool running = false;
    bool result_ready = false;
    bool show_popup = false;
    bool open_succeeded = false;
    std::string target_path;
    std::string progress_message;
    std::string result_status;
  };

  struct OfflineImageFrameLoadTask {
    std::mutex mutex;
    std::thread worker;
    bool running = false;
    bool preview_ready = false;
    bool result_ready = false;
    bool show_popup = false;
    int target_index = -1;
    std::string target_path;
    std::string progress_message;
    cv::Mat frame_bgr;
    cv::Mat preview_bgr;
    DetectionResult detection;
    cv::Size image_size;
    bool found = false;
    std::string result_status;
  };

  struct OfflinePathPickerTask {
    std::mutex mutex;
    std::thread worker;
    bool running = false;
    bool result_ready = false;
    bool show_popup = false;
    bool pick_directory = false;
    bool selection_made = false;
    OfflinePathPickerTarget target = OfflinePathPickerTarget::None;
    std::string current_value;
    std::string progress_message;
    std::string selected_path;
    std::string error_message;
  };

  struct OfflineDetectionTask {
    std::mutex mutex;
    std::condition_variable cv;
    std::thread worker;
    bool stop_requested = false;
    bool busy = false;
    bool has_request = false;
    bool result_ready = false;
    uint64_t pending_request_id = 0;
    uint64_t result_request_id = 0;
    cv::Mat pending_frame_bgr;
    PatternConfig pending_pattern_config;
    bool pending_ir_mode = false;
    bool pending_invert_ir = false;
    std::string progress_message;
    cv::Mat preview_bgr;
    DetectionResult detection;
    cv::Size image_size;
    bool found = false;
    std::string result_status;
  };

  struct CalibrationTask {
    std::mutex mutex;
    std::thread worker;
    bool running = false;
    bool result_ready = false;
    bool show_popup = false;
    float progress = 0.0f;
    std::string progress_message;
    CalibrationSession result_session;
  };

  void addCameraSlot();
  void removeCameraSlot(size_t index);
  CameraSlot* activeSlot();
  bool isDeviceInUse(const std::string& id, const CameraSlot* ignore) const;
  int findSlotByDevice(const std::string& id) const;
  void refreshDevices(CameraSlot& slot);
  void updateFrameProcessing();
  void applyTheme();
  void startProcessingWorker(CameraSlot& slot);
  void stopProcessingWorker(CameraSlot& slot);
  void processingWorkerLoop(CameraSlot* slot);
  void startFeatureWorker(CameraSlot& slot);
  void stopFeatureWorker(CameraSlot& slot);
  void clearFeatureCache(CameraSlot& slot);
  void queueFeatureRefresh(CameraSlot& slot);
  void queueFeatureSet(CameraSlot& slot, const FeatureInfo& feature, const FeatureValue& value);
  void queueFeatureCommand(CameraSlot& slot, const FeatureInfo& feature);
  void featureWorkerLoop(CameraSlot* slot);
  void refreshOfflineImageList();
  void updateOfflineImageLoader();
  void stopOfflineImageLoader();
  void drawOfflineImageLoaderPopup();
  void updateOfflineVideoLoader();
  void stopOfflineVideoLoader();
  void drawOfflineVideoLoaderPopup();
  void updateOfflinePathPicker();
  void stopOfflinePathPicker();
  void drawOfflinePathPickerPopup();
  void updateOfflineImageFrameLoader();
  void stopOfflineImageFrameLoader();
  void drawOfflineImageFrameLoaderPopup();
  void updateCalibrationTask();
  void stopCalibrationTask();
  void drawCalibrationPopup();
  void updateOfflineDetectionTask();
  void startOfflineDetectionWorker();
  void stopOfflineDetectionWorker();
  void offlineDetectionWorkerLoop();
  void queueOfflineDetection(const cv::Mat& frame_bgr, const PatternConfig& pattern_config,
                             bool offline_ir_mode, bool offline_invert_ir,
                             const std::string& status_message);
  void invalidateOfflineDetection();
  bool requestOfflinePathPicker(OfflinePathPickerTarget target, bool pick_directory,
                                const std::string& current_value);
  bool requestOfflineImageLoad(int index);
  bool loadOfflineImage(int index);
  bool openOfflineVideo();
  bool stepOfflineVideo();
  bool startCalibrationTask(const cv::Size& image_size, const PatternConfig& pattern_config);
  void processOfflineFrame();
  void drawCalibrationSourceControls();
  void drawCameraSlot(size_t index, CameraSlot& slot);
  void drawConfigDialog(size_t index, CameraSlot& slot);
  void drawCalibrationPanel(CameraSlot* active);
  void drawFeaturePanel(CameraSlot& slot);

  std::vector<std::unique_ptr<CameraSlot>> slots_;
  int active_slot_ = 0;
  float left_panel_width_ = 370.0f;

  PatternConfig pattern_config_;
  mutable std::mutex pattern_mutex_;
  uint64_t pattern_revision_ = 0;
  CalibrationSession calibrator_;
  StereoCalibrationSession stereo_calibrator_;
  CameraModel camera_model_ = CameraModel::Pinhole;
  PinholeDistortionModel pinhole_distortion_model_ = PinholeDistortionModel::PlumbBob;
  bool target_warp_compensation_ = false;
  CalibrationInputMode calibration_input_mode_ = CalibrationInputMode::Live;
  std::string offline_images_dir_ = "calibration_samples";
  OfflineImageEncoding offline_image_encoding_ = OfflineImageEncoding::Standard;
  std::vector<std::string> offline_image_files_;
  int offline_image_index_ = -1;
  OfflineImageLoadTask offline_image_loader_;
  OfflinePathPickerTask offline_path_picker_;
  OfflineImageFrameLoadTask offline_image_frame_loader_;
  OfflineDetectionTask offline_detection_task_;
  CalibrationTask calibration_task_;
  std::string offline_video_path_;
  cv::VideoCapture offline_video_capture_;
  OfflineVideoOpenTask offline_video_loader_;
  bool offline_video_open_ = false;
  int offline_video_frame_index_ = -1;
  cv::Mat offline_frame_bgr_;
  cv::Mat offline_preview_frame_bgr_;
  DetectionResult offline_detection_;
  cv::Size offline_image_size_;
  SampleMetrics offline_metrics_{};
  bool offline_found_ = false;
  OfflineDetectionState offline_detection_state_ = OfflineDetectionState::Idle;
  uint64_t offline_detection_request_id_ = 0;
  bool offline_ir_mode_ = false;
  bool offline_invert_ir_ = false;
  std::string offline_status_;
  uint64_t offline_timestamp_ns_ = 0;
  uint64_t offline_seq_ = 0;
  ImageTexture offline_texture_;
  UiTheme ui_theme_ = UiTheme::Dark;
  bool stereo_mode_enabled_ = false;
  int stereo_left_slot_ = 0;
  int stereo_right_slot_ = 1;
  bool stereo_fix_intrinsics_ = true;
  float stereo_expected_baseline_m_ = 0.0f;
  float stereo_sync_tolerance_ms_ = 20.0f;

  bool auto_add_ = false;
  bool allow_duplicates_ = false;
  bool save_samples_ = false;
  std::string samples_dir_ = "calibration_samples";
  std::string calibration_save_path_ = "camera_intrinsics.yaml";
  std::string stereo_calibration_save_path_ = "stereo_calibration.yaml";
};
