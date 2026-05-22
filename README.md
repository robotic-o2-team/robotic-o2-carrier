# Robotic O₂ Generator Carrier for LTOT Patients

> A quadruped-based robotic system that autonomously follows COPD patients, carries their oxygen generator, and actively manages oxygen tubing to prevent snagging — enabling safe, independent mobility at home.

![Status](https://img.shields.io/badge/status-active%20development-brightgreen)
![Platform](https://img.shields.io/badge/platform-Unitree%20Go2-blue)
![ROS](https://img.shields.io/badge/middleware-ROS%202-orange)
![Simulation](https://img.shields.io/badge/simulation-NVIDIA%20Isaac%20Sim-76b900)

## Project Overview

Patients with chronic obstructive pulmonary disease (COPD) who require long-term oxygen therapy (LTOT) face significant mobility challenges. Conventional oxygen delivery systems restrict movement, create tripping hazards, and reduce quality of life.

**Our solution:** A Unitree Go2 quadruped robot that:
- Follows the patient autonomously through the home
- Avoids obstacles in cluttered indoor environments
- Manages oxygen tubing via a mounted robotic arm to prevent snagging
- Navigates uneven terrain that wheeled robots cannot handle

## System Architecture

```
┌─────────────────────────────────────────────┐
│           Unitree Go2 Quadruped             │
├──────────┬──────────┬───────────────────────┤
│ Dual     │ Hesai    │ Robotic Arm           │
│ RGB-D    │ LiDAR    │ (Cable Management)    │
│ Cameras  │          │                       │
├──────────┴──────────┴───────────────────────┤
│         Sensor Fusion Pipeline              │
│  YOLO Pose + ReID + Point Cloud to Grid    │
├─────────────────────────────────────────────┤
│        PID-based Person Following          │
│  (Patient Following + Obstacle Avoidance)  │
└─────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Robot Platform | Unitree Go2 |
| Perception | Dual RGB-D cameras + Hesai LiDAR |
| Person Detection | YOLOv8 Pose (TensorRT) |
| Person Re-ID | OSNet (TensorRT) |
| LiDAR Processing | HesaiLidar ROS 2.0, point cloud to grid |
| Navigation | PID controllers + ROS 2 Nav2 |
| ROS Bridge | go2_nav_bridge (custom) |
| Simulation | NVIDIA Isaac Sim |
| Software | ROS 2, Python, C++ |
| Cable Mgmt | Lightweight robotic arm (planned) |

## Repository Structure

```
robotic-o2-carrier/
├── README.md
├── docs/
│   ├── references.md            # External repos and resources (links only)
│   ├── lab_dev_notes.md         # Original lab development notes
│   └── abstract-Robotic O2 ...  # Project abstract (PDF)
├── src/
│   ├── perception/              # Sensors, detection, tracking
│   │   ├── camera_capture.py            # RGB-D camera interface
│   │   ├── depth_processor.py           # Depth image processing
│   │   ├── dual_camera_system.py        # Dual-camera stitching system
│   │   ├── dual_camera_centerline_viewer.py
│   │   ├── nfc_gpu.py                   # GPU-accelerated NFC utilities
│   │   ├── pixel_to_3d_api.py           # 2D-to-3D projection API
│   │   ├── trt_inference.py             # TensorRT inference runner
│   │   ├── yolo_pose_inference.py       # YOLOv8 pose detection
│   │   ├── reid_trt_inference.py        # ReID TensorRT inference
│   │   ├── reid_manager.py              # ReID gallery and matching
│   │   ├── reid_augment.py              # ReID data augmentation
│   │   ├── single_person_tracker.py     # Single-target tracker
│   │   ├── vision_target_export.py      # UDP target export (MPPI backend)
│   │   ├── visualization.py             # OpenCV debug overlays
│   │   ├── HesaiLidar_ROS_2.0/          # Hesai LiDAR ROS2 driver + SDK
│   │   ├── hesai_lidar_filter/          # LiDAR point cloud filter node
│   │   └── pointcloud_to_grid_ros2/     # Point cloud → occupancy grid
│   ├── navigation/              # Robot motion and following
│   │   ├── person_follower.py           # PID-based person following logic
│   │   ├── pid_controller.py            # Generic PID controller
│   │   ├── robot_controller.py          # Go2 motion command interface
│   │   ├── go2_nav_bridge/              # ROS2 nav bridge for Go2
│   │   └── person_follow_nav/           # ROS2 person-following launch + config
│   ├── cable_management/        # Robotic arm control (planned)
│   │   └── .gitkeep
│   └── simulation/              # Isaac Sim / Gazebo environments (planned)
│       └── .gitkeep
├── config/
│   ├── hesai_lidar_config.yaml          # Hesai LiDAR parameters
│   ├── nav2_controller.yaml             # Nav2 controller config
│   ├── pointcloud_grid_config.yaml      # Point cloud → grid parameters
│   └── launch/                          # ROS2 launch files
│       ├── follow_sidecar.launch.py
│       ├── follow_sidecar_lidar.launch.py
│       ├── hesai.launch.py
│       ├── filter.launch.py
│       └── ...
├── scripts/
│   ├── main.py                          # Main vision + following entry point
│   ├── args_parser.py                   # CLI argument definitions
│   ├── structured_logging.py            # ECS-format structured logging
│   ├── debug_trace_logger.py            # Frame-level debug trace logger
│   ├── utils.py                         # Shared utility functions
│   ├── sync.sh                          # File sync utility
│   └── docker/                          # Docker build + run scripts
│       ├── Dockerfile
│       ├── Dockerfile_ros2_sidecar
│       ├── start_follow_system.sh
│       └── ...
└── tests/
    ├── test_copyright.py
    ├── test_flake8.py
    └── test_pep257.py
```

## Getting Started

### Prerequisites
- ROS 2 (Humble or later)
- NVIDIA GPU with CUDA (for TensorRT inference)
- Docker (optional, recommended)

### Running with Docker

```bash
# Build the main vision container
cd scripts/docker
bash docker_build.sh

# Run the person-following system
bash docker_run.sh

# Or use the full system launcher
bash start_follow_system.sh
```

### Running Directly

```bash
# Install Python dependencies (see Dockerfiles for full list)
pip install -r requirements.txt  # TBD

# Run the main vision + following loop
python scripts/main.py --follow --follow-backend pid

# With LiDAR (ROS2 sidecar mode)
# See config/launch/follow_sidecar_lidar.launch.py
```

### ROS2 Build

```bash
# From the repo root, build the ROS2 workspace packages
mkdir -p ros2_ws/src
cp -r src/perception/HesaiLidar_ROS_2.0    ros2_ws/src/
cp -r src/perception/hesai_lidar_filter     ros2_ws/src/
cp -r src/perception/pointcloud_to_grid_ros2 ros2_ws/src/
cp -r src/navigation/go2_nav_bridge         ros2_ws/src/
cp -r src/navigation/person_follow_nav      ros2_ws/src/

cd ros2_ws && colcon build
```

## Key Subsystems

### Person Detection and Re-Identification
- YOLOv8 Pose model (TensorRT engine) detects people and extracts body keypoints
- OSNet ReID model maintains identity across occlusions via an appearance gallery
- `SinglePersonTracker` manages the target lock across frames

### Person Following (PID)
- Depth measured via bimodal histogram on the bounding box region
- Forward/backward speed controlled by a PID on depth error
- Rotation controlled by a PID on angular error from camera principal point
- Edge and size penalties prevent chasing artifacts near frame borders

### LiDAR Pipeline
- Hesai LiDAR driver publishes raw point clouds over ROS2
- `hesai_lidar_filter` crops and filters the point cloud
- `pointcloud_to_grid_ros2` converts to a 2D occupancy grid for Nav2

### Navigation Bridge
- `go2_nav_bridge` translates Nav2 velocity commands to Unitree Go2 sport API calls

## External References

See [`docs/references.md`](docs/references.md) for links to:
- Unitree Go2 ROS2 model
- RL in Isaac Sim setup guide
- Staircase RL model (rl_sar)
- Demo video

## Team

| Name | Role |
|------|------|
| **Olga** | Developer |
| **Aibek** | Developer |

**Supervisors:** Juan Ricardo Wilches, Hieu Tran, Temirzhan Mukhambet, Rice Pham, Khanh Quoc Duong, William Kearns, Yu Sun

**Affiliation:** Center for Innovation, Technology and Aging — University of South Florida

## Milestones

| Week | Milestone | Focus |
|------|-----------|-------|
| 1 | Kickoff & Planning | Team intro, project understanding |
| 2 | Environment Setup | Dev environment, simulation setup |
| 3 | Perception Pipeline | RGB-D + LiDAR fusion, patient detection |
| 4 | Navigation & Following | PID following, obstacle avoidance |
| 5 | Cable Management | Robotic arm control, tube management |
| 6 | Integration & Demo | Full system testing, final presentation |

## License

This project is part of the USF CREATE Award research program.

## Acknowledgement

Funded by the USF Collaborative Research Excellence And Translational Efforts (CREATE) Award.
