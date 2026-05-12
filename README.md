# 🤖 Robotic O₂ Generator Carrier for LTOT Patients

> A quadruped-based robotic system that autonomously follows COPD patients, carries their oxygen generator, and actively manages oxygen tubing to prevent snagging — enabling safe, independent mobility at home.

![Status](https://img.shields.io/badge/status-active%20development-brightgreen)
![Platform](https://img.shields.io/badge/platform-quadruped%20robot-blue)
![Simulation](https://img.shields.io/badge/simulation-NVIDIA%20Isaac%20Sim-76b900)

## 📋 Project Overview

Patients with chronic obstructive pulmonary disease (COPD) who require long-term oxygen therapy (LTOT) face significant mobility challenges. Conventional oxygen delivery systems restrict movement, create tripping hazards, and reduce quality of life. Existing robotic carts struggle in cluttered home environments.

**Our solution:** A quadruped robot platform that:
- **Follows the patient** autonomously through the home
- **Avoids obstacles** in cluttered indoor environments
- **Manages oxygen tubing** via a mounted robotic arm to prevent snagging
- **Navigates uneven terrain** that wheeled robots cannot handle

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────┐
│              Quadruped Platform              │
├──────────┬──────────┬───────────────────────┤
│ Dual     │ 3D       │ Robotic Arm           │
│ RGB-D    │ LiDAR    │ (Cable Management)    │
│ Cameras  │          │                       │
├──────────┴──────────┴───────────────────────┤
│         Sensor Fusion Pipeline              │
│    (Unified Spatial Representation)         │
├─────────────────────────────────────────────┤
│     Potential-Field Navigation              │
│  (Patient Following + Obstacle Avoidance)   │
└─────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Robot Platform | Quadruped (Go2 / Spot) |
| Perception | Dual RGB-D cameras + 3D LiDAR |
| Sensor Fusion | Point cloud stitching, depth alignment |
| Navigation | Potential-field trajectory planner |
| Simulation | NVIDIA Isaac Sim |
| Software | ROS 2, Python, C++ |
| Cable Mgmt | Lightweight robotic arm |

## 👥 Team

| Name | Role |
|------|------|
| **Olga** | Developer |
| **Aibek** | Developer |

**Supervisors:** Juan Ricardo Wilches, Hieu Tran, Temirzhan Mukhambet, Rice Pham, Khanh Quoc Duong, William Kearns, Yu Sun

**Affiliation:** Center for Innovation, Technology and Aging — University of South Florida

## 📅 6-Week Hackathon Milestones

| Week | Milestone | Focus |
|------|-----------|-------|
| 1 | **Kickoff & Planning** | Team intro, project understanding, presentation |
| 2 | **Environment Setup** | Dev environment, simulation setup, initial prototyping |
| 3 | **Perception Pipeline** | RGB-D + LiDAR fusion, patient detection |
| 4 | **Navigation & Following** | Potential-field planner, obstacle avoidance |
| 5 | **Cable Management** | Robotic arm control, tube slack management |
| 6 | **Integration & Demo** | Full system testing, final presentation |

## 🚀 Final Deliverables

1. **Working prototype** (simulation or hardware) of patient-following quadruped
2. **Perception pipeline** — fused RGB-D + LiDAR spatial representation
3. **Navigation system** — potential-field trajectory generator
4. **Cable management** — robotic arm tube handling
5. **Documentation** — technical report + demo video
6. **Final presentation** — project results and demo

## 📁 Repository Structure

```
robotic-o2-carrier/
├── README.md
├── docs/                  # Documentation and presentations
├── src/
│   ├── perception/        # RGB-D + LiDAR fusion
│   ├── navigation/        # Potential-field planner
│   ├── cable_management/  # Robotic arm control
│   └── simulation/        # Isaac Sim environments
├── config/                # Robot and sensor configurations
├── tests/                 # Unit and integration tests
└── scripts/               # Utility scripts
```

## 🔧 Getting Started

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/robotic-o2-carrier.git
cd robotic-o2-carrier

# Setup instructions (TBD)
```

## 📄 License

This project is part of the USF CREATE Award research program.

## 🙏 Acknowledgement

This project is funded by the USF Collaborative Research Excellence And Translational Efforts (CREATE) Award.
