# Data Generator

## Overview
`create_environment.py` is a script that creates and manages environments for robotic manipulation and picking simulations using NVIDIA Isaac Sim. This script simulates scenarios where randomly placed objects in a KLT box are removed one by one, collecting camera data and movement analysis data at each step.

## Key Features

### 1. Environment Setup
- **Physics Engine Optimization**: GPU-based physics simulation optimization settings
- **Lighting System**: Natural environment lighting configuration
- **Camera System**: RGB, depth, and segmentation camera systems
- **Table and KLT Box**: Robotic workspace configuration

### 2. Object Generation and Placement
- **YCB Dataset Objects**: Various everyday objects from the YCB dataset
- **Random Placement**: Natural placement of 30-40 objects within the KLT box
- **Physics Simulation**: Stable placement considering gravity and collisions

### 3. Data Collection
- **Camera Information**: Intrinsic and extrinsic parameters
- **Image Data**: RGB images
- **Depth Data**: Distance information
- **Segmentation**: Per-object instance masks
- **Surface Normals**: 3D surface orientation information
- **Contour Data**: Object boundary information

### 4. 6-DOF Movement Analysis
- **Position Changes**: 3D position change measurements
- **Rotation Changes**: Euler angle and quaternion-based rotation changes
- **Movement Thresholds**: Meaningful movement detection (position: 0.0175m, rotation: 0.13rad)
- **Collision Impact Analysis**: Analysis of how lifted objects affect other objects

### 5. Physics-Based Lifting
- **Contact Point Detection**: Automatic grip point detection using camera data
- **Progressive Lifting**: Step-by-step lifting for natural movement
- **Stability Analysis**: Stability assessment during lifting process

## Usage

### Basic Execution
```bash
python create_environment.py
```

### Key Parameters
```bash
python create_environment.py \
    --num_episodes 10 \
    --min_objects 30 \
    --max_objects 40 \
    --lift_height 0.5 \
    --lift_steps 75 \
    --save_file scenario_1 \
    --save_intermediate \
    --headless
```

### Parameter Description
- `--seed`: Random seed setting (default: -1, automatic)
- `--level`: Difficulty level (default: 1)
- `--method`: Methodology setting (default: 'mcts')
- `--min_objects`: Minimum number of objects (default: 30)
- `--max_objects`: Maximum number of objects (default: 40)
- `--num_episodes`: Number of episodes (default: 10)
- `--render`: Enable rendering (default: 1)
- `--save_file`: Save file name (default: 'synario_1')
- `--lift_height`: Lifting height in meters (default: 0.5m)
- `--lift_steps`: Number of lifting steps (default: 75)
- `--steps_per_frame`: Physics steps per frame (default: 2)
- `--save_intermediate`: Enable intermediate state saving
- `--center_point_lifting`: Use center-point based lifting
- `--headless`: Run in headless mode

## Output Data Structure

```
/home/robot/datasets/isaac_sim_data/[save_file]/
├── episode_0/
│   ├── camera_info/          # Camera intrinsic/extrinsic parameters
│   ├── image_data/          # RGB images (.png)
│   ├── depth/               # Depth data (.bin)
│   ├── contour/             # Contour data (.json)
│   ├── normal/              # Surface normal data (.bin)
│   └── action_metadata/     # Action metadata (.json)
├── episode_1/
└── ...
```

### Data File Formats

#### action_metadata_[step].json
```json
{
  "step": 0,
  "timestamp": null,
  "lifted_objects": ["/World/obj/cracker_box_0"],
  "moved_objects": [
    {
      "object_name": "sugar_box_1",
      "position_magnitude": 0.025,
      "orientation_magnitude": 0.15,
      "position_diff": [0.01, 0.02, 0.001],
      "orientation_diff_euler": [5.2, 2.1, 0.8],
      "6dof_vector": [0.01, 0.02, 0.001, 0.09, 0.037, 0.014]
    }
  ],
  "num_objects_remaining": 35,
  "movement_analysis": { ... },
  "additional_info": { ... }
}
```

## Technical Details

### Physics Engine Configuration
- **Solver**: TGS (Temporal Gauss-Seidel)
- **Broadphase**: GPU-based
- **Collision Detection**: GPU-optimized collision stack
- **Simulation Frequency**: 60Hz

### Camera System
- **Resolution**: 1224x1024 pixels
- **Position**: [0.0, 0.61, 1.58]
- **FOV**: Adjustable horizontal/vertical aperture
- **Sensors**: RGB, depth, instance segmentation, surface normals

### Object Stabilization Process
1. **Initial Placement**: Random positioning of objects
2. **Physics Stabilization**: 150 frames of gravity simulation
3. **Damping Application**: Linear/angular velocity damping for stabilization
4. **Boundary Correction**: Repositioning objects within KLT box boundaries
5. **Final Stabilization**: Additional stabilization time

### Grip Point Detection
- **Camera-Based**: Utilizing depth and segmentation information
- **Surface Analysis**: Grip point selection from top 20% depth regions
- **Coordinate Transformation**: Camera coordinate to world coordinate conversion
- **Safety Validation**: Grip point validity verification
