# rosmaster-a1-e2e-vision

Minimal runnable ROS 2 end-to-end visual driving MVP for Yahboom Rosmaster A1 Ackermann control on RDK X5 + Nuwa RGB camera.

## Architecture

- Runtime node (`rosmaster_a1_e2e_vision/e2e_driver_node`)
  - Subscribes to configurable RGB topic (`sensor_msgs/Image`)
  - Inference backend: `torch` checkpoint or `onnx` session
  - Model output: normalized steering `[-1, 1]` and `stop_prob`
  - Runtime gating:
    - external start/stop (`std_msgs/Bool`)
    - safety stop hook (`std_msgs/Bool`, for future depth-stop integration)
    - visual stop threshold
  - Ackermann steering protections:
    - clamp (`max_steering_abs`)
    - low-pass filter (`steering_low_pass_alpha`)
    - rate limit (`steering_rate_limit_per_sec`)
  - Fixed low-speed control when running and safe
- Control adapter abstraction
  - `ackermann_drive_stamped` (default, `ackermann_msgs/AckermannDriveStamped`)
  - `twist` fallback (`geometry_msgs/Twist`)
- Training scripts (`training/`)
  - PilotNet / DAVE-2 style CNN with dual heads
  - Steering regression + stop classification
  - Masked steering loss (only on go samples)
  - Train/val split + checkpoint save

## Repository layout

- `/rosmaster_a1_e2e_vision`: runtime + model code
- `/launch/e2e_driver.launch.py`: launch entry
- `/config/runtime.yaml`: practical default config
- `/training/train.py`: minimal training pipeline
- `/training/export_onnx.py`: optional ONNX export
- `/training/README.md`: dataset schema details

## Prerequisites

- ROS 2 (Humble or compatible)
- Python packages:
  - Runtime: `torch`, `numpy`, `opencv-python`, `cv_bridge`, `onnxruntime` (only for ONNX backend)
  - Training: `torch`, `numpy`, `Pillow`
- ROS messages:
  - default adapter: `ackermann_msgs`
  - alternative adapter: `geometry_msgs`

## Build and run

```bash
cd /tmp/workspace/cowbook/rosmaster-a1-e2e-vision
colcon build --symlink-install
source install/setup.bash
```

Set `model_path` in `/tmp/workspace/cowbook/rosmaster-a1-e2e-vision/config/runtime.yaml`, then run:

```bash
ros2 launch rosmaster_a1_e2e_vision e2e_driver.launch.py
```

Start/stop control:

```bash
ros2 topic pub /e2e/start_stop std_msgs/msg/Bool '{data: true}' -1
ros2 topic pub /e2e/start_stop std_msgs/msg/Bool '{data: false}' -1
```

Safety stop hook:

```bash
ros2 topic pub /e2e/safety_stop std_msgs/msg/Bool '{data: true}' -1
```

## Runtime configuration points

Main fields in `config/runtime.yaml`:

- Topics: `rgb_topic`, `start_stop_topic`, `safety_stop_topic`, `control_topic`
- Control output type: `control_adapter` (`ackermann_drive_stamped` or `twist`)
- Model: `model_backend`, `model_path`, `model_device`
- Stop logic: `visual_stop_threshold`
- Speed: `fixed_speed_mps`
- Ackermann steering behavior:
  - `max_steering_abs`
  - `steering_low_pass_alpha`
  - `steering_rate_limit_per_sec`

## Training steps

1. Prepare dataset CSV and images (see `training/README.md` and `training/sample_dataset.csv`).
2. Train:

```bash
cd /tmp/workspace/cowbook/rosmaster-a1-e2e-vision
python3 training/train.py \
  --csv /path/to/dataset.csv \
  --image-root /path/to \
  --output-dir training/checkpoints \
  --epochs 20
```

3. Runtime checkpoint: `training/checkpoints/model_best.pt`
4. Optional ONNX export:

```bash
python3 training/export_onnx.py \
  --checkpoint training/checkpoints/model_best.pt \
  --output training/checkpoints/model_best.onnx
```

## Dataset recording expectations

Each sample:

- `image_path`: RGB frame
- `steering`: normalized Ackermann steering command in `[-1, 1]`
- `go`: `1` when moving, `0` when stopped

Recommendations:

- Collect balanced go/stop sequences
- Include starts, stops, and corner cases
- Keep first model with fixed low speed; do not learn throttle in MVP

## Ackermann note for training

Ackermann chassis mainly affects steering label consistency and runtime mapping:

- Train in normalized steering space (`[-1, 1]`) independent of servo units.
- Use runtime `max_steering_abs` to map normalized output to actual steering angle.
- Steering smoothing/rate limiting is especially important on Ackermann to avoid oscillation.

## Adapting control topic/message for Yahboom stack

If your local Yahboom integration uses different command message/topic:

1. Change `control_adapter` and `control_topic` in config.
2. If neither existing adapter matches, add a new adapter in `rosmaster_a1_e2e_vision/control_adapter.py`.
3. Keep the runtime node unchanged and switch only adapter selection via config.

## Safety notes for indoor testing

- Start with wheels lifted and verify command signs first.
- Use low speed (`0.08-0.15 m/s`) for first runs.
- Keep external stop topic bound to a physical/operator control path.
- Keep `safety_stop_topic` wired to future depth stop logic before freer driving.
