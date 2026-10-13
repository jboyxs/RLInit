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
4. 建模为 MDP 过程

先实现「固定底盘，末端到达目标点」，定义 `<S, A, P, R, γ>`，再扩展到开门任务。以下数值是初始参数，需根据仿真效果调整。

| 要素 | 到达任务的定义 |
| --- | --- |
| 状态 S | 完整物理状态、目标位置及影响下一步的控制器状态。策略观测先用 `[七个关节角, 七个关节角速度, 末端位置, 目标−末端位置]`，共 20 维；末端取 `pinch_site`。 |
| 动作 A | `a ∈ [-1, 1]^7`，映射为关节目标角 `q_target = clip(q_arm + 0.03a, 下限, 上限)`，写入七个机械臂位置执行器的 `data.ctrl`。限位需同时满足关节和执行器约束；无限位关节不能把默认 `[0, 0]` 当作限位。 |
| 转移 P | MuJoCo 根据状态和控制输入推进物理。每个动作保持 N 个物理步，控制周期为 `N × model.opt.timestep`；例如物理步长 0.002 s、N=10 时为 50 Hz。 |
| 奖励 R | 令动作执行后的末端距离为 `d = ||p_ee − g||₂`，使用 `r = −d − 0.001||a||₂² + 5 × 1[d < 0.02]`：鼓励接近目标、减小动作，到达 2 cm 内给予成功奖励。 |
| 折扣 γ | 训练算法中先设 `γ = 0.99`；控制周期变化时需重新考虑其对应的时间尺度。 |

环境还需明确以下规则：

- **初始化**：加载 `home` 姿态，真正固定底盘并约束夹爪；避免观测遗漏影响运动的自由度或控制器历史。
- **目标**：每回合在可达、无碰撞的小范围内采样，回合内保持不变；可从合法关节姿态的正向运动学结果采样，并过滤碰撞姿态。
- **结束**：`d < 0.02 m` 表示成功，预先定义的失败状态也结束任务（`terminated`）；达到 250 次动作的外部时间限制则截断（`truncated`）。
- **接口**：封装 Gymnasium 环境，`reset(seed=...) → (obs, info)`，`step(action) → (obs, reward, terminated, truncated, info)`；`info` 记录距离和成功标志。接口见 [官方文档](https://gymnasium.farama.org/api/env/)。
- **验收**：先用零动作和随机动作检查固定约束、关节限位、观测和奖励，确认距离奖励随目标接近而提高、成功和超时信号正确，再接入 PPO/SAC。

开门任务随后扩展：开放底盘与夹爪动作，增加门角度/角速度、把手位姿等观测，并定义抓取、开门进度、碰撞惩罚及成功条件。

- [中文建模教程](docs/mujoco_from_zero.md)：局部坐标、运动学树、关节、轮子、夹爪同步与驱动器。
- [完整 MJCF XML](models/mobile_manipulator.xml)
- [运行脚本](scripts/view_robot.py)

```bash
uv sync --extra cpu
# 只显示模型初始姿态：
uv run --no-sync python scripts/view_robot.py
```

### 第一个任务：PPO 训练-先实现固定底盘，末端到达目标点（简单任务验证）


代码位于 `src/rlinit/algorithmhand/`：`reach_env.py` 实现上述到达任务，`PPO.py` 调用 [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) 提供可复用的训练和评估函数，仓库根目录的 `train.py` 是实际运行入口，负责解析参数并调用这些函数。环境从 TidyBot XML 在内存中移除底盘和夹爪自由度，保留七个臂关节；原模型文件不变。第一版只以位置到达为成功条件，尚未加入碰撞失败判定或末端姿态要求。

在仓库根目录运行（以下以 CPU 环境为例；CUDA 环境安装方式见下文）：

```bash
uv sync --extra cpu
# 正式训练；步数会向上取整到完整 rollout，30 万步是初始预算。
uv run --no-sync python train.py --timesteps 300000
# 实时显示训练中的机械臂和绿色目标点。
uv run --no-sync python train.py --timesteps 300000 --render
# 短跑检查训练流程，不代表策略已经学会任务。
uv run --no-sync python train.py --timesteps 256 --n-steps 128 --batch-size 64 --output-dir runs/smoke
# 加载模型评估；增加 --render 可打开查看器观察策略。
uv run --no-sync python train.py --evaluate runs/reach_ppo/ppo_reach.zip --episodes 20
```

也可在 Python 中引用训练函数：

```python
from rlinit.algorithmhand.PPO import train_ppo

checkpoint = train_ppo(total_timesteps=300_000, output_dir="runs/reach_ppo", seed=0)
# train_ppo(..., render=True) 可在训练时打开查看器。
```

输出目录包含最终模型 `ppo_reach.zip`、训练日志 `train.monitor.csv`、配置 `config.json` 和独立评估指标 `metrics.json`（成功率、平均回报、平均最终距离）。每 10000 步或一个 rollout（取较大值）额外评估并保存 `best/best_model.zip`。训练结束自动进行 10 回合独立评估；是否学会任务以评估结果为准。

默认模型路径依赖当前仓库的 `models/` 目录；使用其他位置的 TidyBot 模型时传入 `--model /绝对路径/tidybot.xml`，训练和评估需使用同一模型。

`--render` 需要图形桌面环境，按约 50 Hz 实时显示运动，会降低训练速度；关闭窗口后训练继续，周期评估和训练结束后的自动评估不打开额外窗口。

### 手写 PPO 训练

`tests/train_hand.py` 使用 `PPOhand.py`，通过 `PPO.from_env(env)` 自动读取观测和动作维度。每次收集 `--n-steps` 步后，使用整段 rollout 更新 `--epochs` 轮。

```bash
# 短跑验证；加 --render 可观察训练。
uv run --no-sync python tests/train_hand.py --timesteps 256 --n-steps 128 --epochs 2
# 正式训练。
uv run --no-sync python tests/train_hand.py --timesteps 300000
# 加载手写 PPO 模型并渲染评估。
uv run --no-sync python tests/train_hand.py --evaluate runs/reach_ppohand/ppo_hand.pt --episodes 20 --render
```

默认输出到 `runs/reach_ppohand/`：模型 `ppo_hand.pt`、回合日志 `episodes.csv`、更新损失 `updates.csv`、配置及最终 10 回合独立评估指标。`.pt` 使用独立格式，不能与 Stable-Baselines3 的 `.zip` 互换；短跑只验证流程，不代表收敛。

### 手写 PPO 的两组独立对照

两版均沿用原版环境、奖励、GAE、动作分布和 Actor/Critic 学习率。默认设置的差异如下：

| 版本 | 网络 | 更新方式 | 入口 | 输出目录 |
| --- | --- | --- | --- | --- |
| 原版 | 单层 128，ReLU | 2048 条整批，10 轮 | `tests/train_hand.py` | `runs/reach_ppohand` |
| 仅网络 | 双层 64，Tanh，SB3 式正交权重初始化 | 原版整批，10 轮 | `tests/train_hand_network.py` | `runs/reach_ppohand_network` |
| 仅更新方式 | 原版单层 128，ReLU | 每轮打乱，每批 64 条，10 轮，批内优势归一化 | `tests/train_hand_minibatch.py` | `runs/reach_ppohand_minibatch` |

网络版保留原版 `log_std=-0.5` 和 tanh 动作变换；小批次版保留原版独立优化器和损失。两版不是完整替换为 SB3 PPO，便于比较网络和更新方式各自的影响。`--hidden-dim` 在网络版中指定两个隐藏层各自的宽度，默认均为 64；小批次版可用 `--batch-size` 修改批大小，尾批也参与更新。

```powershell
# 仅更换网络。
uv run --no-sync python tests/train_hand_network.py --timesteps 300000 --seed 0
uv run --no-sync python tests/train_hand_network.py --evaluate runs/reach_ppohand_network/ppo_hand.pt --render --episodes 20 --seed 10000
# 仅更换更新方式。
uv run --no-sync python tests/train_hand_minibatch.py --timesteps 300000 --batch-size 64 --seed 0
uv run --no-sync python tests/train_hand_minibatch.py --evaluate runs/reach_ppohand_minibatch/ppo_hand.pt --render --episodes 20 --seed 10000
```

两版均在训练结束后保存 `ppo_hand.pt` 并自动评估 10 回合，输出文件名和日志字段与原版一致。新检查点记录版本，评估时自动创建对应网络；原版旧检查点仍可加载。比较各版时使用相同训练预算和评估 `--seed`，以便目标点一致。

四版（原版手写、网络修改版、小批次修改版、SB3）训练结束后的自动评估统一使用 `训练 seed + 20000` 作为首回合种子，之后每回合递增 1，共评估 10 回合。该规则由 `src/rlinit/algorithmhand/evaluation.py` 统一定义；终端打印实际种子，`config.json` 的 `final_eval_seed` 字段记录该值。例如训练 `--seed 0` 时，自动评估的目标种子为 `20000` 到 `20009`。手动评估仍直接使用命令行 `--seed`。

### 两台设备：CPU / RTX 3060

两台设备共用 `pyproject.toml` 和 `uv.lock`，分别选择互斥的 PyTorch extra。`uv sync` 不检测显卡，应明确指定 `cpu` 或 `cuda`，不要同时启用两者（也不要使用 `--all-extras`）。配置方式见 [uv 官方 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies)。

```bash
# 本机无 NVIDIA GPU：安装 CPU 版 PyTorch。
uv sync --extra cpu
# 另一台 RTX 3060：安装 CUDA 12.8 版 PyTorch（Linux / Windows）。
uv sync --extra cuda
```

完成各自同步后，两台设备使用相同命令。`--no-sync` 保留刚选定的依赖；默认 `--device auto` 在 CUDA 可用时选 GPU，否则选 CPU。

```bash
uv run --no-sync python tests/train_hand.py --timesteps 300000
uv run --no-sync python train.py --timesteps 300000
# 在显卡机器强制使用 CUDA；若不可用会直接报错。
uv run --no-sync python tests/train_hand.py --device cuda --timesteps 300000
# 检查安装的 PyTorch、CUDA 构建版本和运行时可用性。
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

显卡机器需要可运行 CUDA 12.8 的 NVIDIA 驱动；安装 CUDA 版 PyTorch 后仍应以上述检查的 `True` 为准。CPU 机器应显示 CUDA 构建版本为 `None`，可用性为 `False`。可用 `--device cpu` 强制 CPU；训练记录所选设备，评估加载模型时重新选择当前设备，允许两台设备间迁移检查点。

GPU 用于策略和价值网络的计算，当前 MuJoCo 物理步进仍在 CPU；小型 MLP 的 PPO 不保证使用 GPU 更快，应比较实际训练耗时。
### 第二个任务：PPO 训练-移动机械臂开门（final goal）
1. 怎么建模门--直接用现有的model
门是否有现有的各种物理属性建模地都比较好的符合现实物理世界的门
2. 机器人和门同场景
能正常加载、位置合理、没有初始穿插

场景入口为 `models/door_scene.xml`，复用 TidyBot，并引入 `models/objects/door_lock_fixture.xml`。门模型改自下载的 `door_lock.xml`：保留原门板尺寸、质量/惯量和两个被动关节，使用纯色材质并添加落地支撑立柱；原下载文件不变。门框固定，把手约在世界坐标 `[0.720, 0.000, 0.775] m`，位于机器人 `home` 姿态的夹爪前方约 14 cm。底盘、机械臂和夹爪自由度均保留。

```powershell
# 静态查看组合场景，自动加载 home 关键帧。
uv run --no-sync python tests/view_door_scene.py
# 检查默认姿态和 home 的穿插、机器人与门的间距、地面间距及 2 秒物理步进。
uv run --no-sync python tests/view_door_scene.py --check
# 验证并保存 home 姿态截图，不打开窗口。
uv run --no-sync python tests/view_door_scene.py --check --screenshot runs/door_scene/preview.png
```

其他加载代码也应显式调用 `mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)`，然后调用 `mujoco.mj_forward(model, data)`。上述查看脚本仅验证场景装配与初始状态；抓取与开门验证见下节。门轴范围沿用原资产的 `0–0.4 rad`（约 23°），后续若采用更大的开门成功角度，需要调整范围并验证门闩和门框的接触。

3. 抓取与开门的物理验证

`tests/verify_door_opening.py` 使用确定性脚本控制：夹爪绕接近方向转 90°，使两指跨把手厚度闭合；接近把手后用 `170/255` 的夹爪命令抓取，再压把手并沿门轴圆弧拉门。机械臂使用位置执行器和偏置力矩前馈，执行器原有力矩限制仍生效。真实仿真的门关节保持被动，未直接设置门角度、未加焊接约束、未对门施加外力；单独的参考状态用于计算末端目标轨迹。

```powershell
# 完整验证：正常抓取开门 + 不压把手 + 不闭合夹爪，两组失败对照。
uv run --no-sync python tests/verify_door_opening.py
# 实时查看正常动作；之后自动运行无窗口对照。
uv run --no-sync python tests/verify_door_opening.py --render
# 保存动作 GIF、抓取/压把手/开门 PNG、报告和轨迹。
uv run --no-sync python tests/verify_door_opening.py --record
```

结果写入 `runs/door_opening/validation.json` 和 `trajectory.csv`，任何验收项失败时进程返回非零退出码。检查包括双侧指垫接触力、抓取后的保持、开门角度、两组失败对照、每个物理步的有限状态/警告/接触穿透，以及底盘位置保持。正常运行的门角度约 20.08°，拉门和保持阶段 50 Hz 采样均有双侧指垫接触，最大接触穿透约 0.28 mm；不压把手和不闭合夹爪的最终门角度分别约 0.82°、0.60°。穿透验收上限为 1.5 mm，属于 MuJoCo 柔顺接触的数值容差。

这是当前小型实验门上的物理控制验证，尚未训练 PPO，也未验证机器人通行或真实门迁移。本次动作中把手压下约 73°；这是此资产上的控制轨迹，不应直接视作普通房间门所需的把手角度。
4. 建模成MDP 
<S,A,R,P,gama>
S:  机器人关节位置、速度  夹爪实际开合状态  门角度、角速度  把手角度、角速度  控制器中会影响下一步的变量
A:  A = [七个机械臂关节的目标角度增量，夹爪开合命令]
R:  小奖励和大奖励和惩罚 夹爪接近把手  抓取姿态对准  双侧指垫抓住把手  抓住后压下把手  门被进一步打开  达到成功开度  撞门框、撞地面、动作过于剧烈
奖励 =
  接近和对准的奖励
  + 抓取、压把手的奖励
  + 开门进度奖励
  + 成功奖励
  − 不良碰撞和动作惩罚
P,gama

