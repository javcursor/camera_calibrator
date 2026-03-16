#include "camera/OpenCVBackend.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_set>

#ifdef __linux__
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace {
constexpr const char* kGStreamerDeviceId = "__opencv_gstreamer_pipeline__";

bool isGStreamerDeviceId(const std::string& id) {
  return id.rfind(kGStreamerDeviceId, 0) == 0;
}

int extractVideoIndex(const std::string& name) {
  const std::string prefix = "video";
  if (name.rfind(prefix, 0) != 0) return -1;
  if (name.size() == prefix.size()) return -1;
  int value = 0;
  for (size_t i = prefix.size(); i < name.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(name[i]))) return -1;
    value = value * 10 + (name[i] - '0');
  }
  return value;
}

bool parseBoolOption(const std::string& value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return (lower == "1" || lower == "true" || lower == "yes" || lower == "on");
}

#ifdef __linux__
std::string v4l2FieldToString(const __u8* data, size_t max_len) {
  const char* text = reinterpret_cast<const char*>(data);
  size_t len = strnlen(text, max_len);
  return std::string(text, len);
}

bool queryCaptureDeviceInfo(const std::string& path, std::string& out_card, std::string& out_bus) {
  int fd = open(path.c_str(), O_RDWR | O_NONBLOCK);
  if (fd < 0) {
    fd = open(path.c_str(), O_RDONLY | O_NONBLOCK);
  }
  if (fd < 0) return false;

  v4l2_capability cap{};
  const int ret = ioctl(fd, VIDIOC_QUERYCAP, &cap);
  close(fd);
  if (ret < 0) return false;

  const uint32_t caps = cap.device_caps != 0 ? cap.device_caps : cap.capabilities;
  const bool can_capture = (caps & V4L2_CAP_VIDEO_CAPTURE) || (caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE);
  if (!can_capture) return false;

  out_card = v4l2FieldToString(cap.card, sizeof(cap.card));
  out_bus = v4l2FieldToString(cap.bus_info, sizeof(cap.bus_info));
  return true;
}
#endif
}  // namespace

std::vector<CameraInfo> OpenCVBackend::listDevices() {
  std::vector<CameraInfo> devices;
#ifdef __linux__
  namespace fs = std::filesystem;
  std::vector<std::pair<int, fs::path>> nodes;
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator("/dev", ec)) {
    if (ec) break;
    const std::string name = entry.path().filename().string();
    int index = extractVideoIndex(name);
    if (index >= 0) nodes.emplace_back(index, entry.path());
  }

  std::sort(nodes.begin(), nodes.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  std::unordered_set<std::string> seen_physical;
  for (const auto& node : nodes) {
    std::string card_name;
    std::string bus_info;
    if (!queryCaptureDeviceInfo(node.second.string(), card_name, bus_info)) {
      continue;
    }

    std::string key = bus_info.empty() ? node.second.string() : bus_info;
    if (!seen_physical.insert(key).second) {
      continue;
    }

    CameraInfo info;
    info.id = node.second.string();
    info.label = card_name.empty() ? ("V4L2 " + node.second.filename().string())
                                   : (card_name + " (" + node.second.filename().string() + ")");
    devices.push_back(info);
  }
#else
  for (int i = 0; i < 16; ++i) {
    cv::VideoCapture cap(i, cv::CAP_ANY);
    if (cap.isOpened()) {
      CameraInfo info;
      info.id = std::to_string(i);
      info.label = "VideoCapture " + std::to_string(i);
      devices.push_back(info);
      cap.release();
    }
  }
#endif
  return devices;
}

bool OpenCVBackend::open(const CameraInfo& info) {
  status_.clear();
  if (isGStreamerDeviceId(info.id)) {
    if (gstreamer_pipeline_.empty()) {
      status_ = "GStreamer pipeline vacio";
      return false;
    }
    if (!cap_.open(gstreamer_pipeline_, cv::CAP_GSTREAMER)) {
      if (!cap_.open(gstreamer_pipeline_, cv::CAP_ANY)) {
        status_ = "No se pudo abrir pipeline GStreamer";
        return false;
      }
    }
    status_ = "GStreamer abierto";
    return true;
  }

  if (info.id.rfind("/dev/", 0) == 0) {
    if (!cap_.open(info.id, cv::CAP_V4L2)) {
      cap_.open(info.id, cv::CAP_ANY);
    }
    if (!cap_.isOpened()) {
      status_ = "No se pudo abrir dispositivo V4L2";
      return false;
    }
    status_ = "Dispositivo OpenCV abierto";
    return true;
  }

  int index = 0;
  try {
    index = std::stoi(info.id);
  } catch (...) {
    index = 0;
  }
  cap_.open(index, cv::CAP_V4L2);
  if (!cap_.isOpened()) {
    cap_.open(index, cv::CAP_ANY);
  }
  if (!cap_.isOpened()) {
    status_ = "No se pudo abrir VideoCapture index";
    return false;
  }
  status_ = "Dispositivo OpenCV abierto";
  return true;
}

void OpenCVBackend::close() {
  cap_.release();
  status_.clear();
}

bool OpenCVBackend::start() {
  return cap_.isOpened();
}

void OpenCVBackend::stop() {}

bool OpenCVBackend::grab(Frame& out) {
  if (!cap_.isOpened()) return false;
  cv::Mat frame;
  cap_ >> frame;
  if (frame.empty()) return false;

  cv::Mat frame_u8 = frame;
  if (frame_u8.depth() != CV_8U) {
    if (frame_u8.channels() == 1 && frame_u8.depth() == CV_16U) {
      double min_v = 0.0;
      double max_v = 0.0;
      cv::minMaxIdx(frame_u8, &min_v, &max_v);
      if (max_v > min_v + 1.0) {
        const double scale = 255.0 / (max_v - min_v);
        const double shift = -min_v * scale;
        frame_u8.convertTo(frame_u8, CV_8U, scale, shift);
      } else {
        frame_u8.convertTo(frame_u8, CV_8U, 1.0 / 256.0);
      }
    } else {
      frame_u8.convertTo(frame_u8, CV_8U);
    }
  }

  if (frame_u8.channels() == 4) {
    cv::cvtColor(frame_u8, frame_u8, cv::COLOR_BGRA2BGR);
  }

  PixelFormat format = PixelFormat::Unknown;
  if (frame_u8.channels() == 1) {
    format = PixelFormat::Mono8;
  } else if (frame_u8.channels() == 3) {
    format = PixelFormat::BGR8;
  } else {
    return false;
  }

  size_t data_size = frame_u8.total() * frame_u8.elemSize();
  auto buffer = makeFrameBuffer(data_size);
  std::memcpy(buffer.get(), frame_u8.data, data_size);

  out.data = std::move(buffer);
  out.data_size = data_size;
  out.width = frame_u8.cols;
  out.height = frame_u8.rows;
  out.stride = static_cast<int>(frame_u8.step);
  out.format = format;
  out.timestamp_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  return true;
}

bool OpenCVBackend::isOpen() const {
  return cap_.isOpened();
}

bool OpenCVBackend::setOption(const std::string& key, const std::string& value) {
  if (key == "gstreamer_pipeline") {
    gstreamer_pipeline_ = value;
    return true;
  }
  if (key == "gstreamer_enabled") {
    gstreamer_enabled_ = parseBoolOption(value);
    return true;
  }
  return false;
}
