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

`tests/train_hand.py` 使用 `PPOhand.py`，通过 `PPO.from_env(env)` 自动读取观测和动作维度。每次收集 `--n-steps` 步后，先按时间顺序计算一次 GAE 和固定目标，再在每个 epoch 打乱样本，按 `--batch-size` 更新。默认每批 64 条，更新 10 轮；批内优势归一化，单条尾批跳过归一化，损失记录所有小批次的平均值。

```bash
# 短跑验证；加 --render 可观察训练。
uv run --no-sync python tests/train_hand.py --timesteps 256 --n-steps 128 --epochs 2
# 正式训练。
uv run --no-sync python tests/train_hand.py --timesteps 300000
# 加载手写 PPO 模型并渲染评估。
uv run --no-sync python tests/train_hand.py --evaluate runs/reach_ppohand/ppo_hand.pt --episodes 20 --render
```

默认输出到 `runs/reach_ppohand/`：模型 `ppo_hand.pt`、回合日志 `episodes.csv`、更新损失 `updates.csv`、配置及最终 10 回合独立评估指标。`.pt` 使用独立格式，不能与 Stable-Baselines3 的 `.zip` 互换；短跑只验证流程，不代表收敛。

手写 PPO 统一使用 `PPOhand.py`，默认双层 128 ReLU、打乱小批次更新；`--hidden-dim` 设置两层宽度，`--batch-size` 设置批大小。主版旧单层检查点仍可评估。

| 任务 | 手写 PPO 入口 | SB3 基准入口 |
| --- | --- | --- |
| 固定底盘到达 | `tests/train_hand.py` | `train.py` |
| 固定底盘开门 | `tests/trainp2_hand.py` | `tests/trainp2_sb3.py` |

手写 PPO 与 SB3 的训练结束后自动评估统一使用 `训练 seed + 20000` 作为首回合种子，之后每回合递增 1，默认评估 10 回合。该规则由 `src/rlinit/algorithmhand/evaluation.py` 统一定义；终端打印实际种子，`config.json` 的 `final_eval_seed` 字段记录该值。例如训练 `--seed 0` 时，自动评估的目标种子为 `20000` 到 `20009`。手动评估仍直接使用命令行 `--seed`。

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
P:仿真引擎决定
gama:自定义

实现位于 `src/rlinit/algorithmhand/door_env.py`，训练入口为 `tests/trainp2_hand.py`，使用原版 `PPOhand.PPO.from_env(env)`。第一版在内存中移除三个底盘关节及执行器，真正固定底盘，保留七个机械臂、八个夹爪机构关节和两个门关节，共 17 个位置/速度坐标、8 个动作输入；源 XML 不变。

当前默认 Actor 为 `64→128→128→8`、Critic 为 `64→128→128→1`，两个隐藏层均使用 ReLU。`--hidden-dim` 同时设置两层的宽度。新检查点记录 `hidden_depth=2`；已有单层检查点在评估时按原结构恢复，新增的隐藏层需要重新训练。

| 要素 | 当前实现 |
| --- | --- |
| S / 观测 | 64 维：全部 `qpos/qvel`（34）、累计臂目标（7）、上一动作（8）、首次抓取/压把手标记（2）、成功保持计时和回合时间比例（2）、相对抓取位置（3）、相对姿态矩阵前两列（6）、左右指垫法向接触力（2）。位置/速度和力按固定尺度缩放。 |
| A | `[-1,1]^8`：前七维使累计关节目标每次增减至多 `0.03 rad`，并限制到关节及执行器允许范围；无显式限位的关节先使用 `home ± π` 的控制范围。最后一维绝对映射到 `0–170` 夹爪命令。机械臂偏置力矩前馈通过原位置执行器实现，力矩限制仍生效。 |
| R | 进度奖励使用 `γΦ(s_next)−Φ(s)`；`Φ=2×接近度+接近度×姿态对准度+2×双侧抓取×把手进度+10×开门进度`。接近度为 `exp(-距离/0.08)`，把手进度按 `1.3 rad` 归一化，开门进度按成功角度归一化并限制在 `[0,1]`。首次双侧抓取 +2，抓住且压把手到 `1.1 rad` 首次 +3，成功 +50，失败 −20；另有碰撞、动作幅度、动作变化和每步 −0.01 的时间惩罚。真正终止时 `Φ(s_next)=0`，超时截断保留下一状态势函数和价值 bootstrap。 |
| P | MuJoCo，每个动作保持 10 个 `0.002 s` 物理步，控制频率 50 Hz。门和把手保持被动，仅通过物理接触运动。 |
| γ | 默认 `0.999`，通过 `--gamma` 配置，奖励势函数和 PPO 使用同一个值。 |

每回合从 `home` 加少量关节扰动开始。成功条件为门角度至少 `0.30 rad`（约 17.2°），且双侧指垫均有超过 `0.05 N` 的法向接触，连续保持 0.30 秒（按控制周期采样）。最大回合 1000 个动作（20 秒）；严重接触穿透或不良碰撞结束回合并惩罚。门闩与门框的正常接触、夹爪与把手的接触不属于碰撞惩罚。

```powershell
# 完整的环境行为验证：固定约束、真实接触开门成功、PPO 更新和检查点恢复。
uv run --no-sync python -m unittest discover -s tests -p test_door_env.py -v
# 短训练验证流程，不代表已经学会开门。
uv run --no-sync python tests/trainp2_hand.py --timesteps 4096 --n-steps 512 --batch-size 64 --epochs 2 --episodes 3 --output-dir runs/door_ppohand_smoke
# 训练；300000 步是初始预算，是否收敛以独立评估为准。
uv run --no-sync python tests/trainp2_hand.py --timesteps 300000 --batch-size 64 --gamma 0.999
# 实时观察训练。
uv run --no-sync python tests/trainp2_hand.py --timesteps 300000 --render
# 双层 128 的独立实验目录，保留此前单层模型用于比较。
uv run --no-sync python tests/trainp2_hand.py --timesteps 300000 --hidden-dim 128 --gamma 0.999 --output-dir runs/door_ppohand_128x128
# 评估：环境配置随检查点恢复，--seed 指定首个评估种子。
uv run --no-sync python tests/trainp2_hand.py --evaluate runs/door_ppohand/ppo_hand.pt --episodes 10 --seed 20000 --render
```

默认输出到 `runs/door_ppohand/`：`ppo_hand.pt`、`config.json`、`episodes.csv`、`updates.csv` 和 `metrics.json`。每次 PPO 更新后保存检查点；训练结束后按 `训练 seed + 20000` 自动独立评估，默认 10 回合。指标包含成功率、曾抓住把手比例、曾压下把手比例、最终/最大门角度及失败原因。检查点保存批大小、gamma 和完整环境配置，并与到达任务检查点区分格式。当前未加入示范学习或自动课程，完成物理脚本验证并不意味着 PPO 已经学会这一连续动作序列。

5. 使用 Stable-Baselines3 PPO 基准训练

`tests/trainp2_sb3.py` 复用相同 `DoorEnv`，通过 `stable_baselines3.PPO("MlpPolicy", env)` 训练，不指定 `policy_kwargs`。网络、激活函数、初始化、优化器和算法参数全部采用安装的 SB3 默认值；当前为 Actor/Critic 各两层 64、Tanh、正交初始化、Adam，学习率 `3e-4`、rollout 2048、小批次 64、更新 10 轮、`gamma=0.99`、GAE lambda 0.95。环境奖励势函数也同步使用实际 PPO gamma。网络和动作分布等与手写版存在差异，这里用于成熟实现的基准实验。

```powershell
# 使用 SB3 默认算法与网络设置训练。
uv run --no-sync python tests/trainp2_sb3.py --timesteps 300000
# 实时观察训练。
uv run --no-sync python tests/trainp2_sb3.py --timesteps 300000 --render
# 仅覆盖 gamma 为 0.999，网络和其他算法参数仍采用 SB3 默认值。
uv run --no-sync python tests/trainp2_sb3.py --timesteps 300000 --gamma 0.999 --output-dir runs/door_sb3_gamma999
# 短跑：保留所有算法默认值，仅缩短交互预算及评估回合数。
uv run --no-sync python tests/trainp2_sb3.py --timesteps 2048 --episodes 1 --output-dir runs/door_sb3_smoke
# 评估最终模型；从 zip 恢复当时的完整环境配置。
uv run --no-sync python tests/trainp2_sb3.py --evaluate runs/door_sb3/ppo_door.zip --episodes 10 --seed 20000 --render
# 评估周期评估中平均回报最高的模型。
uv run --no-sync python tests/trainp2_sb3.py --evaluate runs/door_sb3/best/best_model.zip --episodes 10 --seed 20000 --render
```

默认输出到 `runs/door_sb3/`：最终模型 `ppo_door.zip`、`config.json`、`train.monitor.csv`、`logs/progress.csv`、`eval/evaluations.npz`、`best/best_model.zip` 和最终独立评估 `metrics.json`。周期评估默认每 10000 步进行 10 回合，不打开额外窗口；最佳模型按平均回报选择。训练日志含 SB3 的 KL、clip fraction、entropy loss、explained variance 等指标。训练结束后的评估首种子同样为 `训练 seed + 20000`。SB3 会将训练步数向上取整到完整 rollout，配置文件记录请求步数和实际步数。

`--n-steps`、`--batch-size`、`--epochs`、`--gamma` 只有显式传入时才覆盖 SB3 默认值。`--max-steps`、`--action-scale` 和 `--initial-noise` 是任务环境参数。评估时这些训练参数从模型恢复；只需指定检查点、种子、回合数和是否渲染。`.zip` 与手写版 `.pt` 使用不同的加载入口。

### 使用 SB3 SAC 训练开门任务

`tests/trainp2_sac.py` 复用原版 `DoorEnv`（64 维观测、8 维连续动作），从 home 姿态训练完整开门任务。采用 SB3 默认 SAC：Actor 和两个 Q 网络各为两层 256 ReLU，学习率 `3e-4`、经验池容量 100 万、随机探索 100 步、小批次 256、每采样一步更新一次、`tau=0.005`、`gamma=0.99`、自动学习熵系数。环境奖励势函数同步采用实际 SAC gamma；可用 `--gamma 0.999` 调整。

```powershell
# 正式训练；现有依赖已包含 SAC，无需增加安装包。
uv run --no-sync python tests/trainp2_sac.py --timesteps 300000
# 训练时实时观察，CPU / CUDA 自动选择；也可指定 --device cpu 或 --device cuda。
uv run --no-sync python tests/trainp2_sac.py --timesteps 300000 --render
# 短跑验证：降低探索步数，确保包含 SAC 梯度更新和周期评估。
uv run --no-sync python tests/trainp2_sac.py --timesteps 128 --learning-starts 16 --buffer-size 256 --batch-size 16 --max-steps 8 --eval-freq 64 --episodes 1 --device cpu --output-dir runs/door_sac_smoke
# 评估最终模型；可替换为 runs/door_sac/best/best_model.zip。
uv run --no-sync python tests/trainp2_sac.py --evaluate runs/door_sac/sac_door.zip --episodes 10 --seed 20000 --render
# 自动检查梯度更新、损失有限性、检查点恢复及确定性评估。
uv run --no-sync python -m unittest discover -s tests -p test_door_sac.py -v
```

默认输出到 `runs/door_sac/`：最终模型 `sac_door.zip`、配置 `config.json`、回合日志 `train.monitor.csv`、损失日志 `logs/progress.csv` 和最终评估 `metrics.json`。每 10000 步独立评估 10 回合，保存 `eval/evaluations.npz` 和按平均回报选择的 `best/best_model.zip`；预算不足一个评估周期时不会生成最佳模型。训练结束后的评估首种子为 `训练 seed + 20000`，记录成功率、抓取率、压把手比例及门角度。短跑仅验证流程，不代表学会开门。

`--learning-rate`、`--buffer-size`、`--learning-starts`、`--batch-size`、`--train-freq`、`--gradient-steps`、`--gamma`、`--tau` 和 `--ent-coef` 显式传入时覆盖 SB3 默认值；`--ent-coef` 支持 `auto`、`auto_0.1` 或固定非负数。`--train-freq` 按环境步计，`--gradient-steps -1` 表示每收集一步做一次更新，实际步数可能向上取整到采样周期。`--max-steps`、`--action-scale`、`--initial-noise` 配置训练环境。评估从 `.zip` 内恢复环境参数，不依赖旁边的 JSON；跨机器可用 `--model` 指定场景 XML。模型用于推理与评估，未另存经验池；SAC 模型需通过本入口加载。

SAC 单环境版和并行版默认每回合最多执行 **1000 步**（50 Hz 下为 20 秒仿真时间），可用 `--max-steps` 覆盖。命令行与未显式传入环境配置的 `train()` 使用相同默认值；评估已有检查点时仍恢复其中保存的回合上限。

### 使用当前状态直接奖励训练 SAC

`tests/trainp2_sac_direct.py` 复用 `trainp2_sac.py` 的 SAC 默认参数、网络和训练/评估流程，使用 `src/rlinit/algorithmhand/door_direct_env.py`。奖励计算方式参照 `door_reference_env.py`，直接按动作执行后的状态计分；位置项采用两侧指垫到把手表面的距离之和，对齐项使用距离衰减的正奖励，其余特征和权重参照 `door_env.py`，不计算 `gamma × Phi(s_next) − Phi(s)`：

```text
d = norm(上爪内侧中心 - 把手上表面中心) + norm(下爪内侧中心 - 把手下表面中心)
distance = -distance_weight × d
orientation = orientation_weight × alignment × exp(-d / 0.08)
press   = clip(把手角度 / 1.3, 0, 1)
opening = clip(门角度 / success_angle, 0, 1)
r_dense = distance + orientation + 2 × grasp × press + 10 × opening
r       = r_dense + 抓取里程碑 + 压把手里程碑 + 成功奖励 + 各项惩罚
```

正确抓取姿态下，上爪对应 `left_pad`、下爪对应 `right_pad`，按此固定配对，不随世界高度交换身份。每侧内表面中心为 `pad1/pad2` 两个接触盒各自局部 `-Y` 表面中心的均值。把手上下目标点取横杆的局部 `+Z/-Z` 表面中心，位于 `door_handle_body` 局部坐标 `[0.075, -0.10, +0.02]` 和 `[0.075, -0.10, -0.02]`，随门及把手转动，不添加原抓取目标的接近偏移。两段距离均在世界坐标中计算，单位为米；位置惩罚为距离之和乘以 `-distance_weight`，默认系数以环境构造函数为准。

`info["distance"]` 及监控日志记录新的距离之和，`upper_finger_distance_m/lower_finger_distance_m` 分别记录两段距离，`grasp_target_distance_m` 保留原夹爪中心到抓取目标的距离。64 维观测中的相对抓取目标位置仍沿用 `DoorEnv` 的计算。`alignment` 是原版的姿态对齐度，范围 `[0, 1]`；对齐正奖励使用同一表面距离之和 `d`，对齐度为 0 时为 0，越靠近且越对齐奖励越高。`orientation_weight=1` 时即为 `alignment × exp(-d/0.08)`。`grasp` 表示双侧指尖接触把手。抓取/压把手里程碑每回合分别只奖励一次 `+2/+3`，成功 `+50`、失败 `-20`，碰撞、动作幅度、动作平滑和每步时间惩罚均保留原值。密集项每步直接累加，包括成功、失败或超时的最后一步，奖励值不受 gamma 影响（gamma 仍用于 SAC 折扣）。成功仍要求抓住把手、门角度达到阈值并保持 0.30 秒；观测、动作、物理失败检查和默认 1000 步回合上限沿用原版。

新检查点记录实际的 `distance_weight` 和 `orientation_weight` 系数、`distance_metric=finger_surface_sum`、`position_reward=linear_distance` 和 `orientation_reward=proximity_alignment`。此前 `orientation_reward=orientation_cost` 的检查点仍按保存系数恢复负姿态误差奖励；缺少 `orientation_weight` 时使用原系数 0.25；缺少 `distance_weight` 时使用原系数 5；缺少 `distance_metric` 时恢复原夹爪中心到抓取目标的距离；缺少 `orientation_reward` 时，恢复原对齐项 `alignment × exp(-夹爪中心到抓取目标距离/0.08)`；同时缺少 `position_reward` 时，位置项恢复为 `2 × exp(-夹爪中心到抓取目标距离/0.08)`。各版回报不宜直接比较。

调整新训练的位置和对齐系数，只需修改 `src/rlinit/algorithmhand/door_direct_env.py` 中 `DirectRewardDoorEnv.__init__` 的 `distance_weight` 和 `orientation_weight` 默认值，然后重新启动训练。训练入口、各并行采样进程和周期评估环境直接使用环境默认值，无需同步修改训练脚本；已有检查点评估仍恢复模型保存的系数。

```powershell
# 正式训练，独立输出到 runs/door_sac_direct/。
uv run --no-sync python tests/trainp2_sac_direct.py --timesteps 300000
# 短跑验证，包含 SAC 更新、周期评估和保存。
uv run --no-sync python tests/trainp2_sac_direct.py --timesteps 32 --learning-starts 4 --buffer-size 64 --batch-size 8 --max-steps 4 --eval-freq 16 --episodes 1 --device cpu --output-dir runs/door_sac_direct_smoke
# 评估时从检查点自动恢复直接奖励环境；也可使用原 SAC 评估入口。
uv run --no-sync python tests/trainp2_sac_direct.py --evaluate runs/door_sac_direct/sac_door.zip --episodes 10 --seed 20000 --render
# 验证直接奖励、原版终止规则、单环境/并行训练及检查点恢复。
uv run --no-sync python -m unittest discover -s tests -p test_door_sac_direct.py -v
```

支持原 SAC 的全部命令行参数，也可通过 `--n-envs` 开启并行采样。默认 `gradient_steps=1` 与原单环境脚本一致；并行时可显式传入 `--gradient-steps -1`，令每条采样经验对应一次更新。最终模型、最佳模型均保存独立任务标识，恢复评估无需旁边的 `config.json`。短跑仅验证训练流程，不代表已学会开门。

### 纯 SAC 精细抓握实验

同一入口增加 `--precision-grasp`，使用独立的 `door_sac_grasp_env.py`。默认从带噪声的 home 姿态开始，由 SAC 输出全部八维动作；不调用示范、行为克隆或专家轨迹。该选项把两侧指垫误差等信息加入观测（78 维），加入居中、两侧距离不平衡、过早闭合的成本及双侧接触奖励。动作直接使用原版七个关节增量和一个夹爪命令，不再按距离自动缩放动作或平滑夹爪命令。任务成功与物理失败条件沿用原版。

策略网络直接产生关节动作；已删除笛卡尔动作、雅可比映射和目标位姿伺服。把手几何只用于观测、奖励及成功判定，不生成关节动作。原版 MuJoCo 执行器和动作限幅保留。新任务标识为 `fixed_base_door_precision_joint_sac_v2`；旧精细任务检查点的动作处理不同，不能用于新任务的评估、续训或初始化，新训练必须使用独立目录和经验池。

精细任务的奖励系数集中在 `GraspTrainingConfig`，与原版直接奖励参数分开保存。`--orientation-cost-weight` 可额外启用全局姿态误差成本。检查点保存完整配置，评估自动恢复；改变奖励配置后应使用新的实验目录。各版本回报不可直接比较。

`--grasp-reward-profile strong` 加强距离、对齐、双侧接触、抓握和后续动作的信号，并相应提高成功奖励。接触/抓握/压把手奖励按对齐与居中质量加权，建立对齐的双侧接触后逐渐撤销过早闭合成本；反向扭把手会扣分。抓握子任务不奖励压把手和开门，避免用偏斜碰撞代替正确夹持。`--init-from CHECKPOINT` 可迁移前一轮在线 SAC 的网络权重，使用空经验池与新的优化器；物理、观测与动作映射必须一致。它用于奖励调整或目标阶段切换，不会加载示范，也不会复用旧奖励的经验池。完全续训使用 `--resume` 配合同一实验的经验池，自动恢复已保存的精细任务参数。

`opening` 奖励配置进一步加强压把手/拉门激励，并将所有奖励统一乘以 0.1。改变奖励单位时可用 `--init-critic-scale` 调整迁移时的初始 Q 输出；Actor 权重不变。位置/姿态已经满足闭合准备条件时，强奖励配置也会撤销过早闭合成本，避免策略停在张开的预抓取姿态。

`joint` 配置在 `strong` 基础上增加 `0.5 × 闭合准备度 × (夹爪动作 + 1) / 2`，鼓励在居中且对齐后尝试闭合。它只计算奖励，不替换网络动作。可选 `--gripper-exploration-std 0.8` 在训练中加入与任务几何无关、时间相关的夹爪探索噪声；七个关节动作不叠加该噪声，独立评估不使用探索噪声。两项均属于待验证实验，不代表已学会开门。

`safe_opening` 配置在 `joint` 基础上增加三项有界软惩罚：关节目标与实际位置的最大偏差超过 0.05 rad 后逐渐惩罚（0.20 rad 时为 −2）；接触穿透超过原失败阈值的 20% 后逐渐惩罚（到达阈值时为 −2）；把手角超过 1.2 rad 后惩罚过度按压（1.4 rad 时为 −0.5，最多 −1）。这些只进入奖励，不限制或替换策略动作，不修改物理和成功/失败条件。旧配置的三个系数均为零。该配置用于验证后半程抓握失稳问题，尚无完整开门成功率保证。

`--contact-observation log` 将两项接触力输入从 `F/20` 改为 `log(1+F/20)`，减小少数大接触力对输入尺度的影响，原始力仍用于物理、奖励和日志。配置随模型保存。在线 SAC 权重迁移时，默认要求力输入编码一致；显式 `--init-without-force-inputs` 可清零 Actor、双 Critic 及其目标网络第一层的两列力输入权重，再学习新的编码。零接触时的网络输出保持一致。该选项仅接受本版直接关节动作 SAC 模型，不接受历史控制器模型。

可选的自身状态课程：`tests/collect_door_sac_states.py` 运行已训练的直接关节动作 SAC，保存它实际到达过的、接近且对齐的物理状态；不构造理想姿态，也不提供专家动作。`--training-reset-bank PATH --training-reset-probability 0.7` 将 70% 的训练回合从这些状态开始，其余从 home 开始。执行器命令、上一动作和计时仅作为恢复完整状态的数据，不作为模仿学习目标；后续动作和更新仍完全来自 SAC。训练起点来源写入配置及模型的独立元数据。周期评估、独立评估和渲染始终使用另一个未启用状态库的环境，从原版 noisy home 开始，评估不需要状态库文件。续训若要继续使用状态库，需要再次传入该训练选项。阶段成功不能证明学会从 home 开门。

收集器的 `--phase press` 只保存策略已经实际握住且压下把手的状态。训练时可用 `--expand-training-reset-bank` 动态保存同一在线学习器产生的安全开门进展；它只记录状态，不修改动作。训练的中途起点有一半按开门角度分组从已访问的各段进展均匀采样，另一半仍使用初始状态库；home 起点比例保持不变。状态库保留较早阶段，避免只练习终点附近。进展状态排除失败/终止状态、任何成功计时、接近终点的门角度和即将超时的回合；最终保存各进程的状态库及其来源。评估始终不启用动态状态库。

`--init-from` 可配合 `--replay-buffer` 保留自身在线经验，经验池旁的 `config.json` 必须声明同一直接关节动作 SAC 任务，且物理、观测、奖励和阶段配置与新任务完全一致；含示范标记的记录会被拒绝。精细任务的直接奖励不依赖折扣因子，因此初始化网络或复用这些经验时允许改变 SAC 的 `gamma`；势函数差分任务没有放开这一条件。迁移网络的奖励可不同，但复用经验的奖励必须与目标任务一致，不会把旧奖励错误地混入新任务。

`--policy-warmup-steps N` 在更新网络前，使用当前初始化后的 SAC 策略随机采样动作，收集自己的在线经验，不更新网络，也不提供动作示范。它用于迁移后先保留一批当前策略的真实轨迹，数量计入 `--timesteps`，默认关闭。预热不叠加额外动作噪声；额外探索噪声在正常 SAC 训练阶段生效。

`--n-steps 3` 启用 SB3 自带的多步 SAC 回报（默认 1），用于让成功或失败的奖励更快传到较早的动作。只使用均匀在线回放，与成功回合重采样互斥。载入同任务的自身单步经验时，保留原始动作、观测、奖励及并行/环形存储次序，由 SB3 在采样时计算多步回报；在回合终止、超时和最新未结束的轨迹边界停止累计，避免跨回合拼接。模型、回放和实际步数配置随检查点保存；不修改 SAC 损失公式，也不引入示范。

可选 `--success-replay-fraction 0.25` 只用于完整任务、全程 home 起点的精细 SAC：每批最多 25% 的转移来自学习器自己成功的完整回合，其余均匀采样。SAC 的 Actor/Critic 损失不变，没有行为克隆或动作标签拟合；不是加载专家示范。已有普通经验池按原版成功计时、门角度及双侧接触恢复成功回合标记，超时不算成功，环形覆盖会清除旧标记。参数上限为 0.5，默认关闭；带中途状态课程或示范标记的经验池不允许经此路径标记为 home 成功。独立评估只使用保存的策略网络，不需要经验池。

```powershell
uv run --no-sync python tests/collect_door_sac_states.py runs/door_sac_joint_v2_full/inspect_50000.zip --output runs/door_sac_joint_v2_approach_bank.npz --episodes 30
```

`--training-stage grasp` 把训练目标设为两侧指垫各距对应表面中心不超过 3 cm、对齐度至少 0.9、保持双侧接触 0.2 秒；`press` 要求双侧接触并将把手压到 1.1 rad，保持 0.1 秒。强奖励的压把手目标还要求对齐度至少 0.85、两侧表面误差各不超过 4 cm。这两个子任务每回合仍从 home 开始，`stage_success_rate` 单独记录阶段成功，`success_rate` 始终不把它们算作完整开门。最终必须用 `full` 阶段模型独立验证原版开门条件。阶段间可使用 `--init-from` 迁移 SAC 权重；评估子任务模型时加 `--full-task` 可强制使用原版开门成功条件。

`reach` 是张开夹爪的精确接近子任务：指垫中心到把手中心不超过 2 cm、对齐度至少 0.98、夹爪开口至少 6 cm，保持 0.1 秒。它使用中心距离和旋转角成本，不把接触或压把手计作接近成功。

```powershell
# 从随机策略开始的在线 SAC，四个物理进程，每收集四步更新一次。
uv run --no-sync python tests/trainp2_sac_direct.py --precision-grasp --n-envs 4 --torch-threads 1 --action-scale 0.01 --learning-starts 5000 --gradient-steps 1 --ent-coef auto_0.05 --buffer-size 300000 --timesteps 200000 --episodes 5 --output-dir runs/door_sac_precision --save-replay-buffer
# 加强奖励及全局姿态成本，仍直接输出关节增量，从随机策略开始。
uv run --no-sync python tests/trainp2_sac_direct.py --precision-grasp --grasp-reward-profile strong --orientation-cost-weight 0.5 --n-envs 4 --action-scale 0.01 --learning-starts 5000 --gradient-steps 1 --ent-coef auto_0.05 --buffer-size 300000 --timesteps 200000 --episodes 5 --output-dir runs/door_sac_joint_v2 --save-replay-buffer
# 续训需要保留同样的环境选项；经验池必须来自同一在线实验。
uv run --no-sync python tests/trainp2_sac_direct.py --precision-grasp --n-envs 4 --action-scale 0.01 --learning-starts 5000 --gradient-steps 1 --ent-coef auto_0.05 --buffer-size 300000 --resume runs/door_sac_precision/sac_door.zip --replay-buffer runs/door_sac_precision/replay_buffer.pkl --timesteps 200000 --output-dir runs/door_sac_precision --save-replay-buffer
# 独立评估并记录两侧误差、闭合命令、把手/门角度和接触力。
uv run --no-sync python tests/inspect_door_sac_direct.py runs/door_sac_precision/best/best_model.zip --output-dir runs/door_sac_precision/inspection --episodes 10 --seed 20000 --frames
# 实时观看；环境配置来自检查点，无需再次指定精细任务选项。
uv run --no-sync python tests/trainp2_sac_direct.py --evaluate runs/door_sac_precision/best/best_model.zip --episodes 10 --seed 20000 --render
```

精细任务以完整/阶段成功率为先；完整任务随后比较压把手率、保持接触时的开门进展、连续抓握时间和回报，保存 `best/best_model.zip`。`eval/task_evaluations.json` 记录任务指标，`latest_model.zip` 保存最近评估时的模型，每两万步保存一次 `latest_replay_buffer.pkl`；正常结束另存最终模型和经验池。抓握率仅表示回合内曾发生双侧接触，最终是否学会以独立 home 起点开门评估为准。

历史实验 `runs/door_sac_trained/sac_door.zip` 使用了雅可比位姿控制器和理想抓取位姿先验，其 39/40 开门结果不符合“策略直接学习关节动作”的目标，不能作为本任务完成的证据。保留文件仅作历史记录；当前入口拒绝加载该旧任务模型。

已验证的直接关节动作模型位于 `runs/door_sac_joint_v2_verified/sac_door.zip`。冻结多步回报阶段第 140000 步的检查点后，在未参与训练和选模的种子 70000–70019、71000–71079 上独立评估：完整开门 **97/100**，抓握和压把手均 **100/100**，三次失败均为原版穿透失败。评估从原版带噪声的 home 开始，确定性网络动作直接进入原版关节/夹爪映射，保留门角至少 0.30 rad、双侧抓握并保持 0.30 秒的成功条件。

训练瓶颈、实验对比和最终配置见 [纯 SAC 开门训练总结](docs/SAC_DOOR_TRAINING_SUMMARY.md)。

该模型使用 `joint` 奖励、全局姿态误差成本 0.5、对数接触力观测、SB3 SAC 的三步回报、gamma 0.995、固定熵系数 0.005、学习率 0.0001，以及四个环境下每条采样转移一次更新。网络为默认双层 256 ReLU。它继承此前纯在线 SAC 的网络与同任务自身经验，完整训练过程包含前期采样；140000 只表示多步回报阶段的检查点步数。此实验始终从 home 训练，未使用模仿学习、动作控制辅助、中途起点课程或成功回合重采样。`safe_opening` 软惩罚实验是另一组对照，未用于该成功模型。后期继续训练曾退化，因此复现结果应使用上述固定模型。

`runs/door_sac_joint_v2_verified/evaluation.json` 汇总全部 100 回合，两个原始报告、模型哈希、训练来源配置和实际开门录像 `opening.gif` 保存在同一目录。该目录内的固定模型及验证材料随代码发布；`runs/` 下其余原始训练输出和大经验池按项目规则留在本地。

```powershell
# 观看已验证模型从原始 home 完成开门；CPU 推理与独立评估一致。
uv run --no-sync python tests/trainp2_sac_direct.py --evaluate runs/door_sac_joint_v2_verified/sac_door.zip --model models/door_scene.xml --episodes 1 --seed 70000 --device cpu --render
# 独立复现两批测试，模型不需要任何经验池或状态库文件。
uv run --no-sync python tests/trainp2_sac_direct.py --evaluate runs/door_sac_joint_v2_verified/sac_door.zip --model models/door_scene.xml --episodes 20 --seed 70000 --device cpu
uv run --no-sync python tests/trainp2_sac_direct.py --evaluate runs/door_sac_joint_v2_verified/sac_door.zip --model models/door_scene.xml --episodes 80 --seed 71000 --device cpu
# 使用本地保留的自身在线权重和经验，重跑该多步回报阶段；最佳模型由周期评估选取。
uv run --no-sync python tests/trainp2_sac_direct.py --precision-grasp --grasp-reward-profile joint --orientation-cost-weight 0.5 --contact-observation log --init-from runs/door_sac_joint_v2_multistep_source/model.zip --replay-buffer runs/door_sac_joint_v2_multistep_source/replay_buffer.pkl --gripper-exploration-std 0.3 --action-scale 0.01 --initial-noise 0.02 --max-steps 1000 --timesteps 500000 --n-envs 4 --n-steps 3 --learning-rate 0.0001 --buffer-size 300000 --batch-size 256 --learning-starts 0 --gradient-steps -1 --gamma 0.995 --ent-coef 0.005 --eval-freq 10000 --episodes 5 --seed 67 --device cuda --save-replay-buffer --output-dir runs/door_sac_joint_v2_multistep_repeat
```

### 固定配方的独立 SAC 开门入口

新增 `tests/train_sac_open.py` 和 `src/rlinit/algorithmhand/door_sac_open_env.py`，把成功实验固化为独立的完整任务入口，不调用旧训练脚本。新环境保留原版物理和关节动作，单独定义 78 维观测及成功实验的 `joint` 奖励；不包含子任务、状态库课程或其他实验奖励分支。共同使用已验证的成功率评估回调。

默认训练参数为：三步回报、gamma 0.995、学习率 0.0001、固定熵系数 0.005、batch 256、经验池 300000、四个并行环境、每条转移更新一次、双层 256 ReLU、夹爪训练噪声标准差 0.3、关节步幅 0.01。奖励系数集中修改新环境的 `OpenRewardConfig`，SAC 参数集中在新入口的 `SACOpenSettings`。已有模型的评估与续训恢复其保存配置。

不传模型时从随机网络开始，前 5000 条转移用于随机探索；不会隐式加载已验证模型。历史 97/100 是继承前期纯在线权重和经验后得到的结果，尚不能承诺从零使用该配方就能复现相同成功率。种子、依赖版本、配置、代码及输入文件哈希会写入输出目录。固定种子方便比较，跨硬件/依赖版本不保证逐位一致。

```powershell
# 从零训练；默认 50 万步，此例增加到 100 万步。
uv run --no-sync python tests/train_sac_open.py --timesteps 1000000 --output-dir runs/door_sac_open_scratch
# 使用本地保留的自身在线模型与经验，重跑当时奏效的多步回报阶段。
uv run --no-sync python tests/train_sac_open.py --init-from runs/door_sac_joint_v2_multistep_source/model.zip --replay-buffer runs/door_sac_joint_v2_multistep_source/replay_buffer.pkl --timesteps 500000 --output-dir runs/door_sac_open_initialized
# 恢复新入口的网络、优化器和经验池，追加训练；并行环境数量应与经验池一致。
uv run --no-sync python tests/train_sac_open.py --resume runs/door_sac_open_scratch/final_model.zip --replay-buffer runs/door_sac_open_scratch/replay_buffer.pkl --timesteps 200000 --output-dir runs/door_sac_open_resumed
# 观看本次训练选出的最佳模型。
uv run --no-sync python tests/train_sac_open.py --evaluate runs/door_sac_open_scratch/best/best_model.zip --seed 70000 --episodes 1 --render
# 也可直接评估仓库中的历史成功模型；新环境已在同样 20 个种子上复核为 19/20。
uv run --no-sync python tests/train_sac_open.py --evaluate runs/door_sac_joint_v2_verified/sac_door.zip --seed 70000 --episodes 20
# 短跑集成测试：真实 SAC 更新、保存、初始化、经验池和续训，以及新旧环境一致性。
uv run --no-sync python -m unittest discover -s tests -p test_door_sac_open.py -v
```

每次训练使用新的空输出目录，防止覆盖模型。保存 `final_model.zip`、`best/best_model.zip`、`latest_model.zip`、`replay_buffer.pkl`、`config.json`、周期评估记录及 `evaluation.json`；最后一项评估选中的最佳模型，而非默认选择训练末尾模型。`--no-save-replay` 可关闭经验池保存以节省空间。续训恢复网络、优化器和经验池，仿真环境重新从 home 开始，不等价于保存全部仿真及随机数状态后的逐步精确恢复。评估只需模型与项目代码/场景，无需经验池。

### 使用多个进程并行采样 SAC

`tests/trainp2_sac_parallel.py` 使用 `SubprocVecEnv` 为每个环境启动独立进程，默认 `--n-envs 4`。多个 MuJoCo 环境并行采样，共用主进程的一套 SAC 网络和经验池；网络结构与单环境版相同。Windows 使用 `spawn`，脚本带主入口保护，可直接运行；其他 Python 脚本调用 `train()` 时也应放在 `if __name__ == "__main__":` 内。

```powershell
# 4 个环境进程并行训练；总采样预算为 30 万条，而非每个环境 30 万条。
uv run --no-sync python tests/trainp2_sac_parallel.py --n-envs 4 --timesteps 300000
# 在显卡机器上用 CUDA 更新网络，MuJoCo 仍在 CPU 子进程运行。
uv run --no-sync python tests/trainp2_sac_parallel.py --n-envs 4 --device cuda --timesteps 300000
# 多进程短跑，包含实际梯度更新、周期评估和保存。
uv run --no-sync python tests/trainp2_sac_parallel.py --n-envs 2 --timesteps 128 --learning-starts 16 --buffer-size 256 --batch-size 16 --max-steps 8 --eval-freq 64 --episodes 1 --device cpu --output-dir runs/door_sac_parallel_smoke
# 并行训练不打开查看器；训练完成后单独渲染评估。
uv run --no-sync python tests/trainp2_sac_parallel.py --evaluate runs/door_sac_parallel/sac_door.zip --episodes 10 --seed 20000 --render
# 同时验证单环境 SAC 与真实子进程并行训练。
uv run --no-sync python -m unittest discover -s tests -p "test_door_sac*.py" -v
```

默认 `--gradient-steps -1`，热身结束后每采集一条经验进行一次梯度更新，符合 [SB3 对多环境 off-policy 训练的说明](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#multiprocessing-with-off-policy-algorithms)。例如 4 个环境、`--train-freq 1` 时，每轮采集 4 条经验并更新 4 次；显式设为 `--gradient-steps 1` 时，每轮只更新一次，减少计算量但也改变训练与采样的比例，需比较最终成功率。`--train-freq` 在此按向量环境步计，总步数可能向上取整到 `n_envs × train_freq` 的倍数。

`--torch-threads` 控制主进程的 PyTorch CPU 计算线程数，默认 1；各环境进程的 PyTorch 线程数固定为 1，以减少 CPU 争用。多进程训练时，评估另用一个独立环境进程，训练和周期评估交替执行。可以先比较 2 / 4 个环境；更多进程不保证更快，网络更新和进程通信也会耗时。`config.json` 记录训练耗时 `training_seconds` 与总经验吞吐量 `transitions_per_second`，不含进程启动、最终评估和最终模型保存，包含训练中的周期评估及最佳模型保存；比较速度时请保持总步数、更新比例、批大小、线程数及评估设置一致。

默认输出目录是 `runs/door_sac_parallel/`，每个采样环境单独写入 `workers/<编号>.monitor.csv`，其余模型与指标文件与单环境版一致。周期评估按总采样条数计，不能整除环境数时向上取整，并记录 `effective_eval_freq`；每个环境首种子为 `seed + 编号`。两种入口生成的 SAC 检查点可互相加载评估。

### 参考 `reference/door.py` 的 SAC 奖励实验

新增 `src/rlinit/algorithmhand/door_reference_env.py` 和 `tests/trainp2_sac_reference.py`，把参考文件 `Door.reward()` 的奖励计算应用到当前 TidyBot 开门环境，无需安装 robosuite。默认开启密集奖励（对应参考中的 `reward_shaping=True`）：

- 成功开门：`reward = 1.0`，不再叠加接近和转动奖励。
- 尚未成功：`reward = 0.25 × (1 − tanh(10 × d)) + clip(0.25 × abs(handle_angle / (π/2)), −0.25, 0.25)`。
- `d` 为 `pinch_site` 到 `door_handle_site` 的直线距离，单位米，不使用旧奖励中的抓取目标偏移；当前门带有真实弹簧门闩，对应参考的 `use_latch=True`。
- 全部奖励乘以 `--reward-scale`，默认 1；`--sparse` 切换为参考文件默认的稀疏模式，仅成功时有奖励。

成功判定同参考文件：门角度严格大于 `0.30 rad`，不要求双侧接触或额外保持时间。当前适配环境在成功时立即终止回合；参考文件自身的 episode 生命周期未移植。保留原环境的严重碰撞、穿透、数值异常终止检查和默认 1000 步上限，但不额外添加旧奖励中的时间、动作、碰撞、失败及里程碑奖励项。观测和动作仍为 64 / 8 维，SAC 网络仍采用双层 256 ReLU。

```powershell
# 使用参考密集奖励训练，默认单环境。
uv run --no-sync python tests/trainp2_sac_reference.py --timesteps 300000
# 4 个环境并行采样，保持每条采样经验对应一次梯度更新的比例。
uv run --no-sync python tests/trainp2_sac_reference.py --n-envs 4 --timesteps 300000
# 稀疏奖励实验；可以用 --reward-scale 2 调整整体奖励尺度。
uv run --no-sync python tests/trainp2_sac_reference.py --sparse --output-dir runs/door_sac_reference_sparse
# 短跑检查，包含实际网络更新及周期评估。
uv run --no-sync python tests/trainp2_sac_reference.py --n-envs 2 --timesteps 128 --learning-starts 16 --buffer-size 256 --batch-size 16 --max-steps 8 --eval-freq 64 --episodes 1 --device cpu --output-dir runs/door_sac_reference_smoke
# 根据模型内保存的奖励与环境配置评估；可加 --render 显示。
uv run --no-sync python tests/trainp2_sac_reference.py --evaluate runs/door_sac_reference/sac_door.zip --episodes 10 --seed 20000 --render
# 核对参考公式、真实接触开门、单环境/并行训练和检查点恢复。
uv run --no-sync python -m unittest discover -s tests -p test_door_sac_reference.py -v
```

默认输出到 `runs/door_sac_reference/`，模型、日志、配置和评估文件沿用 SAC 的文件组织方式。模型中保存独立任务标识、奖励模式与缩放系数，最终模型和最佳模型都按此配置恢复评估；原 SAC 入口仍保留其原有奖励。短跑和确定性物理轨迹检查只验证实现，训练效果以独立评估的成功率为准。

`tests/trainp2_sac_reference_large.py` 是独立的大网络入口：Actor 和两个 Q 网络的隐藏层由 `[256, 256]` 扩大到 `[1024, 1024]`，仍使用 ReLU。奖励、环境、SAC 超参数、并行采样和评估方式与参考奖励版相同，默认结果写入 `runs/door_sac_reference_large/`。配置和检查点保存实际网络结构，评估自动恢复。

```powershell
# 大网络训练；单环境时可加 --render 实时查看。
uv run --no-sync python tests/trainp2_sac_reference_large.py --timesteps 300000
# 4 个环境并行采样。
uv run --no-sync python tests/trainp2_sac_reference_large.py --n-envs 4 --timesteps 300000
# 渲染评估大网络模型。
uv run --no-sync python tests/trainp2_sac_reference_large.py --evaluate runs/door_sac_reference_large/sac_door.zip --episodes 10 --seed 20000 --render
```

### 大网络 SAC：固定策略示范预填充经验池

`tests/trainp2_sac_reference_large_demo.py` 在大网络参考奖励版基础上，先利用 `tests/verify_door_opening.py` 的 `DoorTrial.run()` 生成机械臂目标轨迹，再在真实的固定底盘 `ReferenceRewardDoorEnv` 中采集示范。关节目标转换为相对累计控制目标的七维增量动作，夹爪命令转换为第八维动作，裁剪到 `[-1, 1]` 后通过 `env.step()` 执行；64 维观测、奖励、下一观测、成功/失败终止和超时截断全部来自训练环境。不会将原脚本的不同维度物理状态直接塞入经验池。

默认采集 10 回合，初始噪声沿用 `--initial-noise`（默认 0.02），首个示范种子为 `训练 seed + 30000`。用 `--demo-episodes`、`--demo-seed` 和 `--demo-initial-noise` 调整；设 `--demo-initial-noise 0` 可从精确 home 状态采集。失败和超时轨迹也作为真实经验保留，采集报告记录各回合成功、门角度、动作裁剪和失败原因；没有成功示范时打印警告。

采集结束后，在第一次在线交互之前预填充 SAC 的同一个经验池，随后按普通 SAC 均匀采样更新，Actor/两个 Q 网络仍为 `[1024, 1024]`。示范不计入 `--timesteps`，`--learning-starts` 仍表示在线随机探索步数（默认 100；设 0 可直接开始更新）。未增加行为克隆或固定示范采样比例，经验池写满后旧示范会逐步被覆盖，是否学会仍以从 home 独立评估的成功率为准。超时转移保留 bootstrap，成功/失败终止不 bootstrap；并行经验池末行不足时重复开头至多 `n_envs - 1` 条转移补齐，并在配置中记录。容量不足以容纳全部示范时明确报错。

```powershell
# 采集 10 回合示范，再训练 30 万在线步。
uv run --no-sync python tests/trainp2_sac_reference_large_demo.py --demo-episodes 10 --timesteps 300000
# 4 个环境并行训练；示范采集仍在独立的单环境中执行。
uv run --no-sync python tests/trainp2_sac_reference_large_demo.py --n-envs 4 --demo-episodes 10 --timesteps 300000
# 小规模验证：完整采集一次成功示范，再执行少量真实 SAC 更新。
uv run --no-sync python tests/trainp2_sac_reference_large_demo.py --demo-episodes 1 --demo-initial-noise 0 --timesteps 8 --learning-starts 0 --buffer-size 4096 --batch-size 8 --eval-freq 8 --episodes 1 --device cpu --output-dir runs/door_sac_reference_large_demo_smoke
# 加载训练后的自主策略评估，可实时查看。
uv run --no-sync python tests/trainp2_sac_reference_large_demo.py --evaluate runs/door_sac_reference_large_demo/sac_door.zip --episodes 10 --seed 20000 --render
# 验证实际示范转移、终止/超时掩码、并行预填充、网络更新和模型恢复。
uv run --no-sync python -m unittest discover -s tests -p test_door_sac_reference_large_demo.py -v
```

默认输出到 `runs/door_sac_reference_large_demo/`，除原有模型、训练日志和评估结果外，还保存 `demonstrations.npz`（观测、下一观测、动作、奖励、terminated、truncated、回合编号和动作阶段）、`demonstrations.json`（采集设置、环境配置和逐回合结果）及训练结束的 `replay_buffer.pkl`（示范与在线经验）。`config.json` 的 `demonstrations` 字段记录预填充数量、并行补齐数量和文件路径，最佳/最终模型内也保存示范来源信息；评估时不运行固定控制脚本。完整经验池使用 SB3 原生序列化，文件大小随经验池分配容量增长。

`tests/trainp2_sac_reference_small_demo.py` 提供小网络对照：Actor 和两个 Q 网络均使用双层 `[256, 256]` ReLU，示范采集、均匀经验回放、奖励和其余 SAC 默认参数与大网络示范版一致。默认输出到独立目录 `runs/door_sac_reference_small_demo/`，配置和检查点保存实际网络结构。

```powershell
# 小网络示范预填充训练。
uv run --no-sync python tests/trainp2_sac_reference_small_demo.py --demo-episodes 10 --timesteps 300000
# 并行采样；与大网络比较时保持相同 seed、环境数和训练参数。
uv run --no-sync python tests/trainp2_sac_reference_small_demo.py --n-envs 4 --demo-episodes 10 --timesteps 300000
# 加载小网络模型进行渲染评估。
uv run --no-sync python tests/trainp2_sac_reference_small_demo.py --evaluate runs/door_sac_reference_small_demo/sac_door.zip --episodes 10 --seed 20000 --render
```

6. Curriculum Learning：独立课程环境与训练入口

课程环境为 `src/rlinit/algorithmhand/door_curriculum_env.py`，训练脚本为 `tests/train_curriculum.py`。复用原环境的机器人、接触、动作映射和失败检查，使用独立的阶段目标、密集奖励和初始化分布；原 `door_env.py` 及两个原训练入口保持独立。

| 阶段 | 较容易的初始状态 | 本阶段成功条件 | 时间上限 |
| --- | --- | --- | --- |
| `reach` | 已对准、离预抓取点约 4 cm | 到预抓取点 2 cm 内，姿态对准度至少 0.98，保持 0.10 s；夹爪保持张开 | 250 步 / 5 s |
| `grasp` | 已到预抓取点、夹爪张开 | 双侧指垫接触力均超过 0.05 N，保持 0.10 s | 350 步 / 7 s |
| `press` | 已抓住把手、门关闭 | 保持双侧抓取，把手角度至少 1.1 rad，保持 0.10 s | 400 步 / 8 s |
| `open` | 已抓住并压下把手、门关闭 | 保持抓取，门角度至少 0.30 rad，保持 0.30 s | 500 步 / 10 s |
| `full` | 初期混合容易起点与 home，逐步全部从 home 开始 | 完整接近、抓取、压把手、拉门，开门与保持条件同 `open` | 1000 步 / 20 s |

容易起点由机器人位置执行器实际仿真生成，保存关节位置/速度、夹爪机构和控制器目标等状态，仅在 `reset()` 恢复；没有固定抓取、直接驱动门或在策略执行中调用脚本控制器。`home_probability` 表示从原始 home 姿态起步的概率。`reach` 阶段从 0 逐步增加到 1，达标后依次学习 `grasp`、`press`、`open`；随后 `full` 从默认 0.25 逐步增加到 1。70 维观测由原 64 维加五维阶段标记和 home 起步概率组成；动作仍为八维机器人输入。

每步奖励采用当前目标的非正密集成本：`-5×距离`、`-0.25×姿态误差`、未抓住 `-0.5`、未完成压把手/开门的归一化误差；保留动作、平滑性、不良碰撞与时间惩罚。子任务成功一次奖励 +20，完整任务成功 +50，失败 -50。后续阶段松开把手也不会移除压把手和开门成本。密集成本用于鼓励尽快完成当前目标，评估同时记录失败原因与回合长度，不能仅凭平均回报判断是否学会。

SB3 课程训练的 Actor/Critic 隐藏层均扩大到 `[256, 256, 256]`，`--hidden-dim` 设置每层宽度，`--hidden-depth` 设置隐藏层数；激活函数仍为 Tanh，初始化、优化器和算法参数沿用 SB3 默认设置，默认 gamma=0.99。启动时打印实际网络结构，配置与检查点记录该结构，评估按检查点恢复。此前双层 64 的检查点仍可评估；新网络需要重新启动训练，可用单独输出目录保存。每次独立评估默认 10 回合，连续两次成功率至少 80% 才升级；步数达到上限不会强制升级。两个渐进阶段的 home 概率每次增加 0.25。只有 `full` 且全部从 home 起步的阶段也达标，才标记课程完成。训练结束始终另行从 home 评估完整任务，容易起点的子任务成功不计作完整任务成功。

```powershell
# 课程训练：默认总预算 100 万步，评估后自动升级。
uv run --no-sync python tests/train_curriculum.py --timesteps 1000000
# 三层 256 的单独实验目录。
uv run --no-sync python tests/train_curriculum.py --timesteps 1000000 --hidden-dim 256 --hidden-depth 3 --device cpu --output-dir runs/door_curriculum_256x3
# 实时观察当前课程阶段。
uv run --no-sync python tests/train_curriculum.py --timesteps 1000000 --render
# 查看检查点对应的当前课程阶段。
uv run --no-sync python tests/train_curriculum.py --evaluate runs/door_curriculum/ppo_curriculum.zip --episodes 10 --seed 20000 --render
# 严格验证完整开门：所有回合从 home 开始。
uv run --no-sync python tests/train_curriculum.py --evaluate runs/door_curriculum/ppo_curriculum.zip --full-task --episodes 10 --seed 20000 --render
# 验证每个子任务的物理可解性、课程升级与检查点恢复。
uv run --no-sync python -m unittest discover -s tests -p test_door_curriculum.py -v
```

结果写入 `runs/door_curriculum/`：`ppo_curriculum.zip`、阶段最佳模型 `best/`、`curriculum.csv`、各阶段 Monitor 日志、`logs/progress.csv`、阶段评估 JSON、`config.json` 和最终从 home 评估的 `metrics.json`。`curriculum.csv` 记录实际步数、阶段、home 比例、阶段成功率和升级结果；配置记录当前阶段与是否完成课程。阶段最佳模型先按成功率、再按平均回报选择。

可通过 `--eval-freq`、`--success-threshold`、`--patience`、`--home-step` 调整课程升级规则；`--stage` 和 `--home-probability` 可用于单阶段实验，`--full-task` 用于严格完整任务评估。课程检查点包含环境和课程配置，观测维度与原 64 维环境不同，需用此入口评估。课程能增加成功经历的可获得性，完整开门是否学会仍以最终从 home 的成功率为准。

7. 使用手写 PPOhand 进行相同课程训练

`tests/train_curriculum_hand.py` 使用 `PPOhand.PPO.from_env(env)`，复用上述 `DoorCurriculumEnv`；两种 PPO 共用 `src/rlinit/algorithmhand/door_curriculum.py` 中的课程升级规则和确定性评估。阶段、奖励、70 维观测、8 维动作、容易起点、成功条件以及默认的 80% / 连续两次 / home 每次增加 0.25 均相同。

手写版继续使用 Actor/Critic 各两层 128 ReLU、tanh 高斯策略、Actor 学习率 `3e-4`、Critic 学习率 `1e-3`，每个 rollout 按时间顺序计算 GAE，再打乱顺序、小批次更新。默认 rollout 2048、批大小 64、更新 10 轮；新课程入口默认 `gamma=0.99`，与 SB3 课程一致，`--gamma 0.999` 可覆盖。课程实现位于独立训练入口，原 `PPOhand.py` 和 `tests/trainp2_hand.py` 的行为不变。

课程评估在一次完整 rollout 更新结束后进行，因此默认约每 10240 步评估一次；最后一个 rollout 按剩余预算缩短。升级时清空已经消费的 rollout 并重置环境，保留同一组网络和优化器状态。阶段切换或预算结束时尚未完成的训练回合会在 `episodes.csv` 的 `cutoff` 字段标记，独立评估均执行完整回合。

```powershell
# 手写 PPO 的课程训练。
uv run --no-sync python tests/train_curriculum_hand.py --timesteps 1000000
# 实时观察训练。
uv run --no-sync python tests/train_curriculum_hand.py --timesteps 1000000 --render
# 评估保存的当前课程阶段。
uv run --no-sync python tests/train_curriculum_hand.py --evaluate runs/door_curriculum_hand/ppo_curriculum_hand.pt --episodes 10 --seed 20000 --render
# 所有回合从 home 开始，评估完整开门。
uv run --no-sync python tests/train_curriculum_hand.py --evaluate runs/door_curriculum_hand/ppo_curriculum_hand.pt --full-task --episodes 10 --seed 20000 --render
# 验证手写 PPO 更新、检查点恢复、阶段切换及与 SB3 评估口径一致。
uv run --no-sync python -m unittest discover -s tests -p test_door_curriculum_hand.py -v
```

默认输出到 `runs/door_curriculum_hand/`：`ppo_curriculum_hand.pt`、各阶段最佳模型 `best/*.pt`、`episodes.csv`、`updates.csv`、`curriculum.csv`、阶段评估 JSON、`config.json` 和从 home 评估完整任务的 `metrics.json`。检查点保存网络、优化器、环境、课程配置与升级进度；当前入口支持训练和评估。手写课程 `.pt`、SB3 课程 `.zip` 和原 64 维任务 `.pt` 分别用各自入口加载。

8. 改用off policy的方法

### SAC + 相同课程奖励与初始情景

`tests/train_curriculum_sac.py` 以 `tests/trainp2_sac.py` 为 SAC 基准，复用 `tests/train_curriculum_hand.py` 的 `DoorCurriculumEnv`、`CurriculumProgress` 和独立确定性评估。Actor/两个 Q 网络继续使用 SAC 默认双层 `[256, 256]` ReLU；学习率 `3e-4`、经验池容量 100 万、随机探索 100 步、batch 256、每采样一步更新一次、gamma=0.99、tau=0.005、自动熵系数均沿用原 SAC 默认值。原 SAC 的超参数覆盖选项继续有效。

环境使用相同的 `reach → grasp → press → open → full` 阶段、容易起点、密集成本和成功/失败奖励。观测为课程环境的 70 维，动作仍为 8 维；`reach` 阶段保持夹爪张开。默认每 10000 步独立评估 10 回合，连续两次成功率至少 80% 才推进；home 概率按 0.25 增加。总预算默认为原 SAC 的 30 万步，可用 `--timesteps` 增加。

评估和升级在完整 SAC 采样/更新结束后执行，设置更大的 `--train-freq` 时，实际采样步数可向上取整。切换阶段或 home 比例时重置环境，保留同一组网络、优化器及经验池；每条转移的观测都包含其原来的阶段和 home 比例，下一观测也保留原阶段，不会拼接新阶段的 reset 状态。`--learning-starts` 按全局在线步数计，仅执行一次初始随机探索。容易起点仅用于 reset，不预填充示范经验。

```powershell
# SAC 课程训练，默认从 reach 阶段开始。
uv run --no-sync python tests/train_curriculum_sac.py --timesteps 300000
# 延长预算并实时观察当前阶段。
uv run --no-sync python tests/train_curriculum_sac.py --timesteps 1000000 --render
# 短跑验证实际 SAC 更新和课程评估。
uv run --no-sync python tests/train_curriculum_sac.py --timesteps 24 --learning-starts 4 --buffer-size 128 --batch-size 8 --max-steps 8 --eval-freq 12 --episodes 1 --device cpu --output-dir runs/door_curriculum_sac_smoke
# 评估检查点所处阶段。
uv run --no-sync python tests/train_curriculum_sac.py --evaluate runs/door_curriculum_sac/sac_curriculum.zip --episodes 10 --seed 20000 --render
# 严格评估从 home 开始的完整开门。
uv run --no-sync python tests/train_curriculum_sac.py --evaluate runs/door_curriculum_sac/sac_curriculum.zip --full-task --episodes 10 --seed 20000 --render
# 检查 SAC 默认参数、真实更新、阶段切换、经验池和模型恢复。
uv run --no-sync python -m unittest discover -s tests -p test_door_curriculum_sac.py -v
```

默认输出到 `runs/door_curriculum_sac/`：`sac_curriculum.zip`、各阶段最佳模型 `best/*.zip`、各阶段 Monitor 日志、`curriculum.csv`、`logs/progress.csv`、阶段评估 JSON、`config.json` 和最终从 home 评估的 `metrics.json`。最佳模型按当前阶段成功率、再按平均回报选取。检查点保存独立 SAC 课程格式、环境和升级进度，支持 `--model` 指定场景位置，不依赖旁边的配置文件。用本入口评估课程模型；原 64 维 SAC、PPO 课程 `.zip`、手写 PPO `.pt` 各用其原入口。短跑和子任务成功不代表已经学会完整开门。
