### 路线（边做边改）
MuJoCo
│
├── MJCF / MjSpec      <geom mass="0"/>

│      ↓
│   搭建机器人和场景
│      <geom mass="0"/>

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
现在的想法有机器人-正确行走、果蝇-完成各种飞行任务、鸟-完成各种飞行任务、3自由度串联机械臂 + 两指夹爪+底座四轮完成开门关门这种contact-rich的任务
现在偏向的是最后一个
2. 从零搭建移动机械臂(已严肃放弃，东西太多了，直接使用google给出的stanford_tidybot模型)
3. 实现任务目标
先实现固定底盘，末端到达目标点（简单任务验证）
移动机械臂开门（final goal）
4. 建模为MDP过程


      

- [中文建模教程](docs/mujoco_from_zero.md)：局部坐标、运动学树、关节、轮子、夹爪同步与驱动器。
- [完整 MJCF XML](models/mobile_manipulator.xml)
- [运行脚本](scripts/view_robot.py)

```bash
uv sync
# 只显示模型初始姿态：
uv run python scripts/view_robot.py
```
