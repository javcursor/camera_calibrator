#include "camera/BackendFactory.hpp"

#include "camera/AravisBackend.hpp"
#include "camera/GenTLBackend.hpp"
#include "camera/OpenCVBackend.hpp"

std::vector<BackendType> availableBackends() {
  std::vector<BackendType> backends;
#ifdef HAVE_ARAVIS
  backends.push_back(BackendType::Aravis);
#endif
#ifdef HAVE_GENTL
  backends.push_back(BackendType::GenTL);
#endif
  backends.push_back(BackendType::OpenCV);
  backends.push_back(BackendType::OpenCVGStreamer);
  return backends;
}

std::string backendLabel(BackendType type) {
  switch (type) {
    case BackendType::Aravis:
      return "Aravis (GenICam/GigE)";
    case BackendType::GenTL:
      return "GenTL (.cti)";
    case BackendType::OpenCV:
      return "OpenCV VideoCapture";
    case BackendType::OpenCVGStreamer:
      return "OpenCV GStreamer";
    default:
      return "Unknown";
  }
}

std::unique_ptr<ICameraBackend> createBackend(BackendType type) {
  switch (type) {
    case BackendType::Aravis:
#ifdef HAVE_ARAVIS
      return std::make_unique<AravisBackend>();
#else
      return nullptr;
#endif
    case BackendType::GenTL:
#ifdef HAVE_GENTL
      return std::make_unique<GenTLBackend>();
#else
      return nullptr;
#endif
    case BackendType::OpenCV:
    case BackendType::OpenCVGStreamer:
      return std::make_unique<OpenCVBackend>();
    default:
      return nullptr;
  }
}
