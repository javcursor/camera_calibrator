# Third-Party Notices

This project (`camera_calibrator`) is licensed under MIT.
Third-party libraries used by the application keep their own licenses.

## Main Dependencies

| Component | Usage in this project | License |
| --- | --- | --- |
| Dear ImGui | UI framework (fetched by CMake or system copy) | MIT |
| GLFW | Window/context/input | zlib/libpng |
| OpenCV | Computer vision, calibration, image/video I/O | BSD-3-Clause-style OpenCV license |
| OpenGL | Rendering API (system library) | Platform/vendor-specific |
| Threads (`pthread`) | System threading | System/runtime license |

## Optional Dependencies

| Component | When used | License |
| --- | --- | --- |
| Aravis | GenICam/GigE Vision backend (`HAVE_ARAVIS`) | LGPL-2.1-or-later |
| GLib/GObject/GIO (transitive via Aravis) | When Aravis backend is enabled | LGPL-2.1-or-later |
| GenTL producer (`.cti`) | When using GenTL backend | Vendor-specific (often proprietary) |

## Redistribution Notes

1. MIT for this repository does not replace third-party licenses.
2. If you redistribute binaries, include:
   - `LICENSE` (project MIT license)
   - This `THIRD_PARTY_NOTICES.md`
   - Applicable third-party license texts/notices
3. If distributed builds include Aravis/GLib (LGPL), comply with LGPL terms (notices, license text, and replacement/relink rights, especially if static linking is used).
4. OpenCV distributions can include additional third-party notices. Preserve the OpenCV license bundle available in installations such as:
   - `<opencv-prefix>/share/licenses/opencv4/`
5. If OpenCV is built with `OPENCV_ENABLE_NONFREE=ON`, usage/redistribution may involve extra patent or licensing constraints depending on selected modules.

## How This Maps to This Project

- Project source files in this repository: MIT (see `LICENSE`).
- Linked libraries at build/runtime: their own licenses still apply.
- Optional backends can change compliance requirements for a distributed binary.
