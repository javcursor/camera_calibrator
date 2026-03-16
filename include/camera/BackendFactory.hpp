#pragma once

#include "camera/ICameraBackend.hpp"

#include <memory>
#include <string>
#include <vector>

std::vector<BackendType> availableBackends();
std::string backendLabel(BackendType type);
std::unique_ptr<ICameraBackend> createBackend(BackendType type);
