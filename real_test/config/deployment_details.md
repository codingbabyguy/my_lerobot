3) deployment_config.json 中文说明对照表
文件位置：real_test/config/deployment_config.json

project / dataset / checkpoint
配置项	含义	你现在的值建议
project.name	本次实验名	自定义即可
dataset.repo_id	数据集标识	保持和训练一致
dataset.root	数据集本地路径	必须存在
dataset.meta_info_path	数据集 info.json 路径	建议与 root 对齐
dataset.meta_stats_path	数据集 stats.json 路径	建议与 root 对齐
checkpoint.pretrained_model_path	推理模型路径	必须是可用 checkpoint 的 pretrained_model 目录
control（控制主循环）
配置项	含义	建议
control.target_hz	目标循环频率	先 10Hz
control.execution_mode	执行模式（serial 串行）	保持 serial
control.action_wait_timeout_s	单步动作完成等待超时	2~5 秒按实机调
control.keyboard_enabled	启用键盘控制	true
control.max_steps	最大步数，0 为不限	调试可设 100
control.latency_log_csv	时延日志输出	保持
control.action_log_csv	动作日志输出	保持
estop（急停）
配置项	含义	建议
estop.enabled	启用文件急停	true
estop.trigger_file	急停触发文件	保持 /tmp/lerobot_estop
action_schema（动作语义）
配置项	含义	建议
action_schema.names	动作维度顺序	必须与训练数据一致，勿改顺序
action_schema.coordinate_frame	坐标系	manual_relative_frame
action_schema.position_unit	位置单位	meter
action_schema.rotation_representation	旋转表示	rot6d
action_schema.gripper_unit	夹爪单位	normalized
safety（安全限制）
配置项	含义	建议
safety.action_bounds.*	各维硬限幅	可保留数据集统计范围
safety.workspace_bounds	工作空间硬边界	真机务必按工位收紧
safety.max_xyz_speed_mps	末端速度上限	先小后大（你现在偏保守，合理）
safety.max_rot_delta_rad	姿态步进上限	真机先保守
safety.max_gripper_delta_per_step	夹爪单步变化上限	真机先保守
offline_replay / real_test_protocol
配置项	含义
offline_replay.*	离线回放检查参数与输出位置
real_test_protocol.*	空载/简化任务协议回合数和日志位置
robot_adapter（Realman 适配）
配置项	含义	建议
robot_adapter.name	适配器类型	german_arm
robot_adapter.dry_run	是否虚拟运行	真机前改 false
config.sdk_src_path	RM SDK Python 源码路径	必须正确
config.host / config.port	控制器地址	必须可达
config.level	登录权限等级	按控制器账号
config.thread_mode	SDK 线程模式	保持当前
config.run_mode	机械臂运行模式	按 RM 实机要求
config.timeout_ms	SDK通信超时	网络差可适当增大
config.movep_canfd_*	CANFD 下发参数	初期保持保守
config.gripper_*	夹爪映射和控制参数	按夹爪实测标定
config.completion_poll_dt_s	完成判定轮询周期	0.02~0.05
config.completion_pose_tol_m	到位位置容差	实机可适当放宽
config.completion_rot_tol_rad	到位姿态容差	实机可适当放宽
config.manual_origin	采集标定参考系原点	来自 calibration_params.npz
config.manual_rotation	采集标定参考系旋转矩阵	来自 calibration_params.npz
config.lock_work_tool_frame	连接时检查 RM work/tool frame	true
config.expected_work_frame_names	允许的 work frame 名称列表	先用 check_rm_frames.py 实测后填写
config.expected_tool_frame_names	允许的 tool frame 名称列表	先用 check_rm_frames.py 实测后填写
config.workspace_clip_in_adapter	是否在适配器做 base-frame workspace 裁剪	建议 false（由 safety 在策略坐标系裁剪）
config.camera_api	相机接口后端	按当前 OpenCV 构建选择；你这台机器更建议 auto
config.camera_stream_url	网络视频流地址（可选）	无采集卡时可填 GoPro/OBS 输出 URL，优先级高于 camera_device_path
config.camera_device_path	稳定视频设备路径	优先用 /dev/v4l/by-id/...video-index0
config.camera_index	本机相机索引	普通 USB 摄像头可用；有 device_path 时可忽略
config.camera_resolution	采集分辨率	建议和 GoPro/采集卡输出一致
config.camera_fps	采集帧率	建议和任务频率成整数倍
config.camera_buffer_size	OpenCV 采集缓冲	建议 1~2
config.camera_fourcc	采集格式	常用 MJPG
config.camera_rotation_deg	画面旋转	0/90/180/270
config.camera_flip_horizontal	水平翻转	按实际安装方向
config.camera_flip_vertical	垂直翻转	按实际安装方向
config.camera_crop_ratio	额外中心裁剪比例	默认 1.0；若想更贴近训练视野可试 0.65~0.85
config.image_shape	送入策略的图像尺寸	代码会先保宽高比缩放并中心裁剪，再 resize 到该尺寸
config.initial_gripper	初始夹爪归一化值	按任务初态设置
