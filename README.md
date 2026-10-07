### 路线（边做边该）
MuJoCo
│
├── MJCF / MjSpec
│      ↓
│   搭建机器人和场景
│
├── Physics Simulation
│      ↓
│ qpos / qvel / sensor / actuator
│
├── Gymnasium Environment
│      ↓
│ observation
│ action
│ reward
│ terminated / truncated
│
├── RL Algorithm
│      ↓
│ PPO / SAC / SAC+HER
│
├── Training
│      ↓
│ rollout → buffer → update → rollout...
│
├── Evaluation
│      ↓
│ success rate
│ return
│ sample efficiency
│
└── Robustness
       ↓
 observation noise
 mass/friction
 object position
 
1. 建模对象的选择和实现的控制任务
现在的想法有机器人-正确行走、果蝇-完成各种飞行任务、鸟-完成各种飞行任务、自由度串联机械臂 + 两指夹爪+底座四轮完成开门关门这种contact-rich的任务
现在偏向的是最后一个
2. 从零搭建移动机械臂

已提供一个教学模型：6 自由度串联机械臂 + 两指平行夹爪 + 四轮滑移转向底座。
尺寸、质量和控制参数为学习用示例，当前场景包含机器人和地面，开关门任务后续添加。

- [中文建模教程](docs/mujoco_from_zero.md)：局部坐标、运动学树、关节、轮子、夹爪同步与驱动器。
- [完整 MJCF XML](models/mobile_manipulator.xml)
- [运行脚本](scripts/view_robot.py)

```bash
uv sync
uv run python scripts/view_robot.py --demo
# 不打开窗口，检查 XML 和仿真：
uv run python scripts/view_robot.py --headless --seconds 10 --demo
```
