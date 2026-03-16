#pragma once

#include "camera/ICameraBackend.hpp"

#ifdef HAVE_ARAVIS
#include <arv.h>
#endif

#include <unordered_map>
#include <unordered_set>

class AravisBackend : public ICameraBackend {
 public:
  BackendType type() const override { return BackendType::Aravis; }
  std::string name() const override { return "Aravis"; }

  std::vector<CameraInfo> listDevices() override;
  bool open(const CameraInfo& info) override;
  void close() override;
  bool start() override;
  void stop() override;
  bool grab(Frame& out) override;
  bool isOpen() const override;

  bool supportsFeatures() const override { return true; }
  std::vector<FeatureInfo> listFeatures() override;
  bool getFeatureValue(const FeatureInfo& info, FeatureValue& out) override;
  bool setFeatureValue(const FeatureInfo& info, const FeatureValue& value) override;
  bool executeCommand(const FeatureInfo& info) override;

  std::string status() const override { return status_; }

 private:
#ifdef HAVE_ARAVIS
  void collectFeatures(ArvDomNode* node, std::vector<FeatureInfo>& out,
                       std::unordered_set<ArvDomNode*>& visited,
                       std::unordered_set<std::string>& seen_ids);
  ArvGcFeatureNode* findFeatureNode(const std::string& id) const;

  ArvCamera* camera_ = nullptr;
  ArvStream* stream_ = nullptr;
  ArvDevice* device_ = nullptr;
  ArvGc* genicam_ = nullptr;
  size_t payload_ = 0;
  std::unordered_map<std::string, ArvGcFeatureNode*> feature_nodes_;
#endif

  std::string status_;
};
