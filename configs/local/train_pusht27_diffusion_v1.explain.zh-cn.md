# train_pusht27_diffusion_v1.json 字段说明

这份配置是给你当前数据集设计的“初级可跑版”，目标是先稳定跑通，再做提速和调参。

数据情况依据：
- total_episodes = 27
- total_frames = 19237
- fps = 10
- observation.image shape = [224, 224, 3]
- observation.state/action shape = [10]

以上来自你的数据元信息文件：
- /home/icrlab/tactile_work_Wy/data/pushT-1.p-20/batch_8/lerobot_v3/meta/info.json

---

## 顶层训练配置

- output_dir
  - 训练输出目录（checkpoint、日志、train_config.json 都会写到这里）。
  - 要求：目录不能已存在；已存在且 resume=false 会报错。

- job_name
  - 训练任务名，用于日志和输出目录标识。

- resume
  - 是否从已有训练继续。
  - 你当前设置 false，表示从头训练。

- seed
  - 随机种子，控制可复现性。

- cudnn_deterministic
  - true 时更可复现但更慢；false 时更快。

- num_workers
  - DataLoader 进程数。
  - 服务器较忙或 CPU 紧张时可改成 2。

- batch_size
  - 单步 batch 大小。
  - 你当前先用 8，兼顾显存和稳定。

- steps
  - 总训练步数。
  - 27 条 episode 的简单任务，先用 15000 做首轮基线。

- eval_freq
  - 评估频率。0 表示不在训练中做环境评估（你当前是离线数据训练，先关掉最稳）。

- log_freq
  - 日志打印频率（每多少步打印一次）。

- save_checkpoint
  - 是否保存 checkpoint。

- save_freq
  - 每多少步存一次。
  - 小数据建议保存频繁一些，便于回滚与挑选最优。

- use_policy_training_preset
  - 是否使用策略内置的 optimizer/scheduler 预设。
  - true 时会用 DiffusionConfig 里的默认优化器配置。

- rename_map
  - 观测键名映射（当数据键名与策略期望不一致时使用）。
  - 你的键名目前是标准格式，先留空 {}。

---

## dataset 配置

- dataset.repo_id
  - 数据集标识符。
  - 本地训练时主要用于标识和日志，不一定要真实 Hub 仓库名。

- dataset.root
  - 本地 v3 数据集根目录。
  - 你现在指向：/home/icrlab/tactile_work_Wy/data/pushT-1.p-20/batch_8/lerobot_v3

- dataset.use_imagenet_stats
  - 图像归一化是否使用 ImageNet 统计。
  - Diffusion+ResNet 常用 true。

- dataset.streaming
  - 是否用流式读取（通常用于 Hub 大数据）。
  - 你是本地单数据集，false 更直接。

---

## policy（Diffusion）配置

- policy.type = diffusion
  - 指定使用 Diffusion Policy。

- policy.device = cuda
  - 使用 GPU 训练。
  - 若当前机器无可用 CUDA，框架会自动回退到可用设备（会变慢）。

- policy.use_amp = true
  - 开启混合精度，通常能降低显存和提升吞吐。

- policy.push_to_hub = false
  - 本地实验阶段先不上传 HF。

- policy.n_obs_steps = 2
  - 输入观测窗口长度（当前帧 + 过去一帧）。

- policy.horizon = 16
  - 模型每次预测动作窗口长度。

- policy.n_action_steps = 8
  - 每次推理实际执行的动作步数。

- policy.vision_backbone = resnet18
  - 视觉 backbone，轻量稳定，适合起步实验。

- policy.crop_is_random = true
  - 训练时随机裁剪增强，提高泛化。

- policy.noise_scheduler_type = DDPM
  - 扩散噪声调度类型，默认稳定选项。

- policy.num_train_timesteps = 100
  - 扩散训练时间步数，默认值。

- policy.optimizer_lr = 1e-4
  - 初始学习率。

- policy.scheduler_name = cosine
  - 学习率调度策略。

- policy.scheduler_warmup_steps = 500
  - warmup 步数。

---

## wandb 配置

- wandb.enable = false
  - 先关闭外部记录，简化首轮实验。

- wandb.project
  - 仅在启用 wandb 时生效。

---

## 推荐启动命令

在 lerobot 环境中执行：

python -m lerobot.scripts.lerobot_train \
  --config_path /home/icrlab/tactile_work_Wy/lerobot/configs/local/train_pusht27_diffusion_v1.json

或：

lerobot-train \
  --config_path /home/icrlab/tactile_work_Wy/lerobot/configs/local/train_pusht27_diffusion_v1.json

---

## 你下一步最可能会改的 4 个参数

- batch_size
  - 显存不够就降到 4；显存富余可尝试 12 或 16。

- steps
  - 想快速看 loss 曲线可先 5000；想拿更稳定结果可到 20000。

- num_workers
  - IO 慢可提高到 6-8；CPU 紧张则降到 2。

- policy.use_amp
  - 若出现混合精度相关不稳定，可临时改成 false。
