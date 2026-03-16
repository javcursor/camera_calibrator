#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

enum class FeatureType {
  Integer,
  Float,
  Boolean,
  Enumeration,
  String,
  Command,
  Category,
  Unknown,
};

struct FeatureInfo {
  std::string id;
  std::string display_name;
  FeatureType type = FeatureType::Unknown;
  bool readable = false;
  bool writable = false;
  double min = 0.0;
  double max = 0.0;
  std::vector<std::string> enum_entries;
  void* backend_handle = nullptr;  // opaque to the UI
};

struct FeatureValue {
  std::variant<int64_t, double, bool, std::string> value;
};
