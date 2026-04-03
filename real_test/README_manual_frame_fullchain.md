# Manual Frame 全链路说明（采集→后处理→训练→推理）

本文档对应双仓流程：

- 采集/后处理：`/Users/wangyi/Vscode/vscode_python_work/tactile_work/tactile_work/Dual-exumi`
- 训练/推理：`/Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot`

目标：全链路保持 `manual_relative_frame` 绝对位姿语义，避免坐标漂移导致的单向漂移与关节顶限。

## 1. 坐标契约（必须一致）

采集端（manual frame）：

- `p_manual = R_manual^T (p_base - o_manual)`
- `R_manual_obj = R_manual^T R_base_obj`

后处理（AR_03）：

- 默认只做时间对齐，不改空间语义。
- 输出 `aligned_arcap_poses.json`，其中 `pose` 仍是 `manual_relative_frame`。

训练转换（LeRobot v3）：

- `observation.state = [x,y,z,rot6d_0..5,gripper]`
- `action = observation.state`（绝对目标，不是 delta）

推理适配（manual -> RM base）：

- `p_base = o_manual + R_manual p_manual`
- `R_base_obj = R_manual R_manual_obj`

关键约束：

- `action_schema.coordinate_frame == manual_relative_frame`
- `robot_adapter.config.policy_frame == manual_relative_frame`
- `startup_pose.map_startup_to_policy_origin == false`（默认）

## 2. 关键命令

### 2.1 采集（Dual-exumi）

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/tactile_work/Dual-exumi
python do_umi.py
```

采集键位：

- `w` 开始 batch
- `u` 记录区间起点
- `j` 记录区间终点
- `e` 结束 batch
- `d` 删除并重采当前 batch
- `q` 退出

规则：第一个 `u-j` 为 ArUco 标定段，后续 `u-j` 为可用 episode。

### 2.2 后处理流水线（Dual-exumi）

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/tactile_work/Dual-exumi
python scripts_slam_pipeline/run_arcap_pipeline.py data/<session_name>
```

如需兼容历史 Flexiv 语义（不推荐）：

```bash
python scripts_slam_pipeline/run_arcap_pipeline.py data/<session_name> --legacy_flexiv_transform
```

### 2.3 转 LeRobot v3（Dual-exumi）

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/tactile_work/Dual-exumi
python scripts/convert_session_to_lerobot_dp.py \
  --session_dir /Users/wangyi/Vscode/vscode_python_work/tactile_work/tactile_work/Dual-exumi/data/<session_name>/batch_1 \
  --output_dir /Users/wangyi/Vscode/vscode_python_work/tactile_work/data/<dataset_name>/lerobot_v3 \
  --repo_id local/<dataset_name> \
  --task <task_name>
```

转换报告输出：

- `reports/conversion_summary.json`
- `reports/deployment_manual_frame_template.json`
- `reports/manual_frame_contract.json`

### 2.4 训练（my_lerobot）

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot
python -m lerobot.scripts.lerobot_train \
  --config_path /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot/configs/local/train_pusht27_diffusion_v1.json
```

### 2.5 生成部署配置（my_lerobot）

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot
python real_test/scripts/build_deployment_config.py \
  --template /Users/wangyi/Vscode/vscode_python_work/tactile_work/data/<dataset_name>/lerobot_v3/reports/deployment_manual_frame_template.json \
  --base-config /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot/real_test/config/deployment_config.json \
  --output /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot/real_test/config/deployment_config.generated.json
```

### 2.6 推理前 frame 校验（my_lerobot）

先在配置里填写：

- `robot_adapter.config.expected_work_frame_names`
- `robot_adapter.config.expected_tool_frame_names`

然后执行：

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot
python real_test/scripts/check_rm_frames.py \
  --config /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot/real_test/config/deployment_config.generated.json \
  --strict
```

### 2.7 干跑 / 真机推理（my_lerobot）

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot
python real_test/scripts/realtime_inference.py \
  --config /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot/real_test/config/deployment_config.generated.json \
  --dry-run
```

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot
python real_test/scripts/realtime_inference.py \
  --config /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot/real_test/config/deployment_config.generated.json
```

## 3. 推理安全链（当前实现）

- `safety.enable_policy_workspace_clip=true`：策略坐标系 workspace 裁剪开启。
- `startup_pose.map_startup_to_policy_origin=false`：禁用 startup 锚点重映射，保持绝对 manual 语义。
- `robot_adapter.config.lock_work_tool_frame=true` + `frame_lock_require_expected_names=true`：frame 锁必须有 expected 名称。
- `robot_adapter.config.runtime_joint_guard.enabled=true`：每步下发前做 IK+软限位/奇异性等关节预检。

## 4. 非真机验证建议

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot
python real_test/scripts/offline_replay_check.py \
  --config /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot/real_test/config/deployment_config.generated.json
```

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot
python real_test/scripts/pseudo_inference_compare.py \
  --config /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot/real_test/config/deployment_config.generated.json
```

### 4.1 PyBullet 起点体检（新增）

先安装依赖：

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot
python -m pip install pybullet matplotlib
```

运行示例：

```bash
cd /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot
python real_test/scripts/pybullet_episode_start_check.py \
  --config /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot/real_test/config/deployment_config.generated.json \
  --urdf-path /path/to/rm_65_description/urdf/rm_65_description.urdf \
  --urdf-package-roots rm_65_description=/path/to/rm_65_description \
  --ee-link-name Link6 \
  --episodes 0,1,2,3,4 \
  --steps 0,1,2,3 \
  --output-dir /Users/wangyi/Vscode/vscode_python_work/tactile_work/my_lerobot/real_test/results/start_pose_check
```

核心产物：

- `start_pose_check.csv`：每个候选起点的 IK 误差/关节裕量
- `start_pose_check_summary.json`：`usable_ratio`、`ik_ok_ratio`、`soft_margin_ok_ratio`
