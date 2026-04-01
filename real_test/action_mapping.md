# 10D Action -> Real Robot Command Mapping

This document freezes the action contract between policy output and the real robot command interface.

## Source Of Truth

- Dataset metadata: /home/icrlab/tactile_work_Wy/data/pushT-1.p-20/batch_8/lerobot_v3/meta/info.json
- Checkpoint package: /home/icrlab/tactile_work_Wy/lerobot/outputs/train/pusht27_diffusion_v2/checkpoints/015000/pretrained_model
- Deployment config: /home/icrlab/tactile_work_Wy/lerobot/real_test/config/deployment_config.json

## Action Order (Exactly 10 Dimensions)

| Index | Name     | Meaning                                | Unit      | Frame      |
|------:|----------|----------------------------------------|-----------|------------|
| 0     | x        | End-effector position X                 | meter     | manual_relative_frame |
| 1     | y        | End-effector position Y                 | meter     | manual_relative_frame |
| 2     | z        | End-effector position Z                 | meter     | manual_relative_frame |
| 3     | rot6d_0  | Rotation 6D first column, element 0     | unitless  | manual_relative_frame |
| 4     | rot6d_1  | Rotation 6D first column, element 1     | unitless  | manual_relative_frame |
| 5     | rot6d_2  | Rotation 6D first column, element 2     | unitless  | manual_relative_frame |
| 6     | rot6d_3  | Rotation 6D second column, element 0    | unitless  | manual_relative_frame |
| 7     | rot6d_4  | Rotation 6D second column, element 1    | unitless  | manual_relative_frame |
| 8     | rot6d_5  | Rotation 6D second column, element 2    | unitless  | manual_relative_frame |
| 9     | gripper  | Gripper open ratio (0 close, 1 open)    | normalized| gripper    |

At runtime, adapter maps this manual frame action into RM base frame pose by:
- `p_base = manual_origin + manual_rotation @ p_manual`
- `R_base = manual_rotation @ R_manual`

## Rotation Convention

- rot6d = [a1, a2], where a1 and a2 are two 3D vectors.
- Runtime conversion to SO(3):
  1. b1 = normalize(a1)
  2. b2 = normalize(a2 - dot(a2, b1) * b1)
  3. b3 = cross(b1, b2)
  4. R = [b1 b2 b3]
- The safety layer in scripts/safety.py performs this orthonormalization before sending commands.

## Mapping To Robot Command Payload

Policy action array is converted to this dictionary before robot API call:

{
  "x": float,
  "y": float,
  "z": float,
  "rot6d_0": float,
  "rot6d_1": float,
  "rot6d_2": float,
  "rot6d_3": float,
  "rot6d_4": float,
  "rot6d_5": float,
  "gripper": float
}

Robot adapter must consume the same keys and must not reorder dimensions.

## Safety Constraints Applied Before Sending

- Hard per-dimension clamp: safety.action_bounds
- Workspace clamp: safety.workspace_bounds for x/y/z
- Cartesian speed cap: safety.max_xyz_speed_mps
- Rotation step cap: safety.max_rot_delta_rad
- Gripper step cap: safety.max_gripper_delta_per_step
- Emergency stop: estop.trigger_file exists -> immediately stop action sending

## Frequency Contract

- Dataset fps = 10
- Real-time loop target_hz = 10
- Every control step logs end-to-end latency and overrun status.

## Deployment Checklist

- Verify `manual_origin/manual_rotation` comes from the same collection calibration.
- Verify RM current work/tool frame matches expected frame names before inference.
- Verify gripper polarity (0 close, 1 open).
- Verify action keys and order unchanged.
- Verify preprocessor/postprocessor files are loaded from the same checkpoint folder.
