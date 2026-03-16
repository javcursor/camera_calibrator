#pragma once

#include "camera/Feature.hpp"
#include "core/Frame.hpp"

#include <string>
#include <vector>

struct CameraInfo {
  std::string id;
  std::string label;
  bool available = true;
};

enum class BackendType {
  OpenCV,
  OpenCVGStreamer,
  Aravis,
  GenTL,
};

class ICameraBackend {
 public:
  virtual ~ICameraBackend() = default;

  virtual BackendType type() const = 0;
  virtual std::string name() const = 0;

  virtual std::vector<CameraInfo> listDevices() = 0;
  virtual bool open(const CameraInfo& info) = 0;
  virtual void close() = 0;
  virtual bool start() = 0;
  virtual void stop() = 0;
  virtual bool grab(Frame& out) = 0;
  virtual bool isOpen() const = 0;

  virtual bool supportsFeatures() const { return false; }
  virtual std::vector<FeatureInfo> listFeatures() { return {}; }
  virtual bool getFeatureValue(const FeatureInfo& info, FeatureValue& out) { (void)info; (void)out; return false; }
  virtual bool setFeatureValue(const FeatureInfo& info, const FeatureValue& value) { (void)info; (void)value; return false; }
  virtual bool executeCommand(const FeatureInfo& info) { (void)info; return false; }

  virtual bool setOption(const std::string& key, const std::string& value) { (void)key; (void)value; return false; }
  virtual std::string status() const { return {}; }
};
