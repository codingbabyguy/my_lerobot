# real_test 使用说明（中文）

本目录用于真机推理验证，当前实现重点是“稳定串行闭环”：

1. 策略输出动作
2. 下发给机械臂执行
3. 等待动作完成
4. 再读状态/图像
5. 再做下一次推理

同时包含安全约束、急停、键盘控制、日志和可视化。

## 1. 目录与脚本作用

- `action_mapping.md`：10维动作定义（x,y,z,rot6d_0~5,gripper）与坐标/单位约定。
- `TEST_PROTOCOL.md`：空载与简化任务测试流程说明。
- `config/deployment_config.json`：统一配置入口（模型、机器人IP、控制参数、安全参数、日志路径）。
- `scripts/realtime_inference.py`：实时推理主循环（串行闭环 + 安全过滤 + 记录日志）。
- `scripts/adapters.py`：机器人适配层（Realman SDK连接、状态读取、动作下发、动作完成判定、急停）。
- `scripts/keyboard_control.py`：终端键盘控制（暂停/继续/急停/退出）。
- `scripts/safety.py`：动作限幅、工作空间限制、速度限制、姿态/夹爪步进限制、文件急停。
- `scripts/offline_replay_check.py`：离线回放检查脚本（用于先验证数据/动作分布）。
- `scripts/protocol_runner.py`：空载/简化任务回合记录。
- `scripts/visualize_realtime.py`：读取实时日志，输出统计 summary 与可视化图。

## 2. 实际运行前你必须配置的内容

请先编辑 `config/deployment_config.json`，至少确认以下字段：

### 2.1 模型与数据

- `dataset.repo_id`
- `dataset.root`
- `checkpoint.pretrained_model_path`

### 2.2 控制模式（当前建议）

- `control.execution_mode`: 建议保持 `serial`
- `control.target_hz`: 建议先用 `10`
- `control.action_wait_timeout_s`: 动作完成等待超时（如 `3.0`）
- `control.keyboard_enabled`: 建议 `true`

### 2.3 机器人连接（真机必须）

- `robot_adapter.name`: `german_arm`
- `robot_adapter.dry_run`: 真机时改为 `false`
- `robot_adapter.config.sdk_src_path`: RM Python SDK 路径
- `robot_adapter.config.host`: 机械臂控制器IP
- `robot_adapter.config.port`: 控制器端口（通常 `8080`）

### 2.4 动作完成判定参数（串行闭环关键）

- `robot_adapter.config.completion_poll_dt_s`
- `robot_adapter.config.completion_pose_tol_m`
- `robot_adapter.config.completion_rot_tol_rad`

### 2.5 相机输入（可选）

- `robot_adapter.config.camera_index`: 填相机编号（如 `0`）
- 不配置或打开失败时会自动回退为占位黑图，不阻塞推理

### 2.6 安全边界（真机前必须收紧）

- `safety.workspace_bounds`
- `safety.max_xyz_speed_mps`
- `safety.max_rot_delta_rad`
- `safety.max_gripper_delta_per_step`

### 2.7 急停

- `estop.enabled`: 建议 `true`
- `estop.trigger_file`: 默认 `/tmp/lerobot_estop`

## 3. 环境准备

推荐使用已配置好的解释器：

```bash
cd /home/icrlab/tactile_work_Wy/lerobot
/home/icrlab/miniforge3/envs/lerobot/bin/python -V
```

可选依赖（用于完整可视化图与USB相机）：

```bash
/home/icrlab/miniforge3/envs/lerobot/bin/python -m pip install matplotlib opencv-python
```

说明：

- `cv2` 用于相机读取。
- `matplotlib` 用于生成 `realtime_dashboard.png`。
- 没有 `matplotlib` 时仍会输出 `realtime_summary.json`。

## 4. 推荐运行顺序（从稳到真机）

## 4.1 离线检查（可选但建议）

```bash
cd /home/icrlab/tactile_work_Wy/lerobot
/home/icrlab/miniforge3/envs/lerobot/bin/python real_test/scripts/offline_replay_check.py \
  --config real_test/config/deployment_config.json
```

输出：

- `real_test/results/offline/offline_summary.json`
- `real_test/results/offline/action_distribution.csv`
- `real_test/results/offline/segment_metrics.csv`

## 4.2 串行闭环 dry-run（先验证逻辑）

```bash
cd /home/icrlab/tactile_work_Wy/lerobot
/home/icrlab/miniforge3/envs/lerobot/bin/python real_test/scripts/realtime_inference.py \
  --config real_test/config/deployment_config.json \
  --dry-run
```

日志输出：

- `real_test/results/realtime_latency.csv`
- `real_test/results/realtime_actions.csv`

终端键盘控制：

- `p`：暂停
- `c`：继续
- `e`：急停并退出
- `q`：正常退出

文件急停（任意终端执行）：

```bash
touch /tmp/lerobot_estop
```

## 4.3 真机运行

先把 `robot_adapter.dry_run` 改成 `false`，再执行：

```bash
cd /home/icrlab/tactile_work_Wy/lerobot
/home/icrlab/miniforge3/envs/lerobot/bin/python real_test/scripts/realtime_inference.py \
  --config real_test/config/deployment_config.json
```

## 4.4 可视化与状态监控

```bash
cd /home/icrlab/tactile_work_Wy/lerobot
/home/icrlab/miniforge3/envs/lerobot/bin/python real_test/scripts/visualize_realtime.py \
  --config real_test/config/deployment_config.json
```

输出：

- `real_test/results/visualization/realtime_summary.json`
- `real_test/results/visualization/realtime_dashboard.png`（若安装了 matplotlib）

新增关键监控字段（在 `realtime_latency.csv` 和 summary 中可见）：

- `wait_ms`：动作完成等待时间
- `action_complete`：动作是否在超时前完成（1/0）
- `key_event`：键盘事件统计
- `camera_source`：图像来源（`camera` 或 `placeholder`）

## 4.5 协议化测试记录（空载/简化任务）

```bash
cd /home/icrlab/tactile_work_Wy/lerobot
/home/icrlab/miniforge3/envs/lerobot/bin/python real_test/scripts/protocol_runner.py \
  --config real_test/config/deployment_config.json
```

输出：

- `real_test/results/real_test_protocol.jsonl`

## 4.6 伪推理对比（不下发真机）

用途：

- 从数据集中取一段观测输入模型，模拟推理。
- 按 chunk 开环对比：每个 chunk 仅使用起点观测，预测一整段动作（默认 8 步）。
- 生成可视化图：
  - 全局时间轴动作对比曲线
  - chunk 内动作对比曲线
  - 末端位姿多维对比（xyz + rot6d）
  - 末端 xyz 轨迹对比
  - 简化 6 轴线条机械臂快照对比（GT vs Pred）

运行示例（按 episode 选片段）：

```bash
cd /home/icrlab/tactile_work_Wy/lerobot
/home/icrlab/miniforge3/envs/lerobot/bin/python real_test/scripts/pseudo_inference_compare.py \
  --config real_test/config/deployment_config.json \
  --episode-index 0 \
  --start-step 0 \
  --num-chunks 25
```

运行示例（按全局索引选片段）：

```bash
cd /home/icrlab/tactile_work_Wy/lerobot
/home/icrlab/miniforge3/envs/lerobot/bin/python real_test/scripts/pseudo_inference_compare.py \
  --config real_test/config/deployment_config.json \
  --global-start 5000 \
  --num-chunks 20 \
  --chunk-size 8 \
  --chunk-stride 8
```

输出文件（默认目录 `real_test/results/pseudo_inference`）：

- `pseudo_compare.csv`：包含 `chunk_id / step_in_chunk / chunk_start / step` 的逐维 GT / Pred / Error
- `pseudo_summary.json`：整体与各维 MAE / RMSE / MAX_ABS
- `compare_actions_global.png`：全局时间轴 10 维动作对比
- `compare_actions_chunked.png`：chunk 内时间轴 10 维动作对比
- `compare_pose_dims_chunked.png`：chunk 内末端位姿多维对比（xyz + rot6d）
- `compare_xyz_trajectory.png`：末端 xyz 轨迹对比
- `compare_simple_arm_snapshots.png`：简化线条6轴机械臂快照

可调参数：

- `--episode-index`：指定从哪个 episode 取数据
- `--start-step`：在该 episode 内从第几步开始
- `--start-offset`：兼容旧参数，等价于 `--start-step`
- `--global-start`：全局步起点（与 `--episode-index` 二选一）
- `--num-chunks`：chunk 数量
- `--chunk-size`：每个 chunk 步数，默认读模型 `n_action_steps`
- `--chunk-stride`：chunk 起点间隔，默认等于 `chunk-size`
- `--num-arm-snapshots`：机械臂快照数
- `--output-dir`：输出目录

说明：

- 该脚本不会连接机械臂，也不会发送动作。
- 如果未安装 `matplotlib`，仍会生成 `csv/json`，但不输出 png 图。

## 5. 常见问题

### 5.1 没有生成 dashboard 图

通常是缺少 `matplotlib`。安装后重跑 `visualize_realtime.py`。

### 5.2 相机一直是 placeholder

请检查：

- `camera_index` 是否正确
- 当前环境是否可导入 `cv2`
- 相机是否被其他进程占用

### 5.3 真机动作不安全/抖动

优先收紧以下参数：

- `workspace_bounds`
- `max_xyz_speed_mps`
- `max_rot_delta_rad`
- `max_gripper_delta_per_step`

### 5.4 动作等待超时（action_complete=0）

可逐步调整：

- 增大 `action_wait_timeout_s`
- 放宽 `completion_pose_tol_m` / `completion_rot_tol_rad`
- 检查机械臂当前模式与轨迹执行状态

## 6. 当前推荐默认策略

- 先 `dry-run` 验证串行链路与日志。
- 再真机空载运行，确认 `action_complete` 长期接近 1。
- 最后进入简化任务，逐步收紧安全参数。
