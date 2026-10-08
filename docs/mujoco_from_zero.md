# 从零手写 MJCF：四轮底座 + 6 自由度机械臂 + 两指夹爪

README 提出了移动机械臂开关门的方向，但没有指定自由度数、尺寸、质量和轮式结构。本示例选择 **6 个转动关节、平行两指夹爪、四轮滑移转向**，先学习机器人本体，再添加门。所有数值都是教学参数，不代表真实设备；六关节也不意味着在所有姿态下都能独立控制末端六维运动。

完整文件在 `models/mobile_manipulator.xml`，静态查看脚本在 `scripts/view_robot.py`。

## 1. 先运行，再逐块改 XML

项目使用 uv，Python 版本沿用项目的 3.14。依赖已登记在 pyproject.toml：

```bash
uv sync
# 打开交互窗口，仅显示 XML 定义的初始姿态：
uv run python scripts/view_robot.py
# 查看其他 XML：
uv run python scripts/view_robot.py --model 你的文件.xml
```

关闭窗口退出。修改 XML 后重新运行脚本。窗口需要可用的桌面显示和 OpenGL。脚本只调用 mj_forward 计算初始几何位置，不推进物理仿真，也不发送动作。

## 2. 认识 XML 中的六个基本概念

| 元素 | 在这个机器人中负责什么 |
| --- | --- |
| `worldbody` | 世界、地面，以及机器人根节点 |
| `body` | 一个刚体和它的局部坐标系；嵌套结构定义父子关系 |
| `joint` | 该刚体相对父刚体允许怎样运动 |
| `geom` | 形状、碰撞、摩擦；本例也通过它推算惯量 |
| `site` | 没有碰撞和质量的参考点，例如夹爪抓取中心 |
| `actuator` | 给关节施加驱动力/力矩，例如位置和速度伺服 |

**`body` 内没有 joint，就与父 body 固定连接。** joint 写在运动的子 body 中，不必像 URDF 那样另写 parent/child。geom 的外观不是关节，给柱子画上去不会让它转动。

采用米、千克、秒；设定 x 向前、y 向左、z 向上，并写 `compiler angle="radian"`。

坐标一定要分清：

- `body pos` 相对于父 body。
- `joint pos/axis`、`geom pos/fromto`、`site pos` 相对于所在 body。
- 子 body 会跟着父 body 平移、转动，不要把每节机械臂的高度都写成世界高度。
- `box size="a b c"` 是三个方向的**半尺寸**；`cylinder size="r h"` 是半径和半高度。
- cylinder 默认沿局部 z 轴；轮子沿 y 轴，所以本例只把轮子 geom 用 `zaxis="0 1 0"` 旋转，轮子 body 坐标系不转。
- `capsule fromto="x1 y1 z1 x2 y2 z2"` 用两个端点定义中轴，`size` 给半径；胶囊的圆头会超过端点。

## 3. 先画运动学树

```text
world
├── floor
└── base [freejoint]
    ├── wheel_fl [hinge, y]
    ├── wheel_fr [hinge, y]
    ├── wheel_rl [hinge, y]
    ├── wheel_rr [hinge, y]
    └── pedestal [固定]
        └── link1 [hinge, z]
            └── link2 [hinge, y]
                └── link3 [hinge, y]
                    └── link4 [hinge, z]
                        └── link5 [hinge, y]
                            └── link6 [hinge, z]
                                └── palm [固定]
                                    ├── finger_left  [slide, +y]
                                    └── finger_right [slide, -y]
```

轮子并列、机械臂嵌套、手指并列。先把这三种关系弄清楚，XML 就是在把这棵树写出来。

## 4. 从地面和一个底座开始

可以先在单独的练习 XML 中只写下面这些：

```xml
<mujoco model="first_base">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom type="plane" size="4 4 0.1"/>
    <body name="base" pos="0 0 0.18">
      <freejoint name="base_free"/>
      <geom type="box" size="0.28 0.18 0.06" mass="15"/>
    </body>
  </worldbody>
</mujoco>
```

底座全尺寸为 0.56 × 0.36 × 0.12 m。这里只有底座时，它会落到地面上，这是正常行为。加轮子后才由轮子支撑。没有 `freejoint` 时底座固定在世界上，即使轮子旋转也不会行驶。

这个最小文件可以用 `uv run python scripts/view_robot.py --model 你的文件.xml` 静态查看，镜头可以用鼠标调整。

## 5. 给底座添加四个轮子

一个轮子的展开写法如下，放进 base 内：

```xml
<body name="wheel_fl" pos="0.20 0.215 -0.075">
  <joint name="wheel_fl_joint" type="hinge" axis="0 1 0"
         limited="false" damping="0.05" armature="0.01"/>
  <geom type="cylinder" size="0.10 0.025" zaxis="0 1 0"
        mass="0.5" friction="0.8 0.01 0.001"/>
</body>
```

轮子半径 0.10 m、宽度 0.05 m，初始轮心世界高度为 `0.18 - 0.075 = 0.105 m`，轮底距地面 5 mm。前/后分别把 x 写成 ±0.20，左/右把 y 写成 ±0.215。轮子需要无限旋转，不能继承机械臂的有限角度范围。

完整文件通过 `<default class="wheel">` 省掉重复属性，再用 `childclass="wheel"` 让轮子 body 内的 joint/geom 使用这些默认值。

这里没有转向关节，靠左右侧轮速差转弯，属于滑移转向。地面接触必须允许侧向滑动，因此不能期待与理想无滑动差速模型完全一致。

## 6. 把单关节扩展成六关节串联臂

先理解肩关节和肘关节这一段：

```xml
<body name="link2" pos="0 0 0.10">
  <joint name="arm_j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
  <geom type="capsule" fromto="0 0 0 0 0 0.24" size="0.025" mass="0.8"/>
  <body name="link3" pos="0 0 0.24">
    <joint name="arm_j3" type="hinge" axis="0 1 0" range="-2 2"/>
    <geom type="capsule" fromto="0 0 0 0 0 0.22" size="0.025" mass="0.6"/>
  </body>
</body>
```

肩关节位于 link2 原点；link2 的中轴长度为 0.24 m；link3 原点放在其末端。转动肩时肘和其后的夹爪一起运动，转动肘只影响肘之后的部分。

本例从根部到末端的轴依次为 **z、y、y、z、y、z**。这是教学布局，初始竖直伸直姿态存在运动学奇异性；后续做逆运动学时选一个弯曲的工作姿态。实际机器人应根据关节轴、连杆尺寸和惯量重建，不能只改颜色。

先让一个关节动，再加第二个，最后扩展到六个。新关节如果不能动，检查是不是少了 joint、控制器引用错了关节、目标超出范围、或几何体互相卡住。

## 7. 两个手指 + 一个开合控制量

两指是 palm 的两个子 body，分别沿 +y 和 -y 滑动：

```xml
<equality>
  <joint joint1="finger_right_joint" joint2="finger_left_joint"
         polycoef="0 1 0 0 0"/>
</equality>
```

这段写在 worldbody 外，让右指位移等于左指位移。因为轴向相反，同样的正位移会让两指对称张开。只给左指添加位置 actuator，通过约束带动右指。

在完整模型中，q=0 时两指内表面还有 0.02 m 间隙，q=0.035 m 时为 0.09 m，因此 **开口宽度 = 0.02 + 2q**。这是可以抓小把手的教学尺寸；想完全合拢，需要一起调整两指初始 y 位置、宽度和行程。equality 是数值求解的软约束，不能保证任何载荷下误差严格为零。

`grasp_site` 是抓取参考点。它的位置由机器人姿态决定，读取方法为：

```python
grasp_position = data.site("grasp_site").xpos.copy()
```

site 本身不会产生接触，实际抓取靠两指 geom 和物体碰撞、摩擦。

## 8. 添加驱动器，把 ctrl 变成可理解的命令

```xml
<actuator>
  <velocity name="drive_fl" joint="wheel_fl_joint" kv="3"
            ctrlrange="-10 10" forcerange="-8 8"/>
  <position name="arm_a2" joint="arm_j2" kp="150" kv="15"
            ctrlrange="-1.5 1.5" forcerange="-40 40"/>
  <position name="gripper" joint="finger_left_joint" kp="400" kv="10"
            ctrlrange="0 0.035" forcerange="-20 20"/>
</actuator>
```

本例使用默认 gear=1：轮子的 ctrl 是 rad/s，机械臂的 ctrl 是目标角度 rad，夹爪的 ctrl 是单指目标位移 m。forcerange 对 hinge 是力矩限制，对 slide 是力限制。

位置伺服近似 `力/力矩 = kp × (目标位置 - 实际位置) - kv × 速度`，再经过力限幅。它不是瞬间把关节移到目标位置；有重力和接触负载时会有误差。先用适中的增益和力限幅，看到抖动先排查接触穿透、质量和步长，再调增益。

后续添加动作时，加载模型并控制一个关节的核心只有四步：

```python
model = mujoco.MjModel.from_xml_path("models/mobile_manipulator.xml")
data = mujoco.MjData(model)
data.ctrl[model.actuator("arm_a2").id] = 0.3
mujoco.mj_step(model, data)
```

加入动作后，在循环中设置 ctrl，然后调用 mj_step，再刷新 viewer。当前查看脚本只显示初始姿态。读取某个关节优先用 `data.joint("arm_j2").qpos[0]`，不要猜数组下标。

### 自由度和控制维度不是同一个数

| 部分 | qpos 数量 | qvel 数量 | actuator 数量 |
| --- | --- | --- | --- |
| 浮动底座 | 7：位置 3 + 四元数 4 | 6 | 0 |
| 四轮 | 4 | 4 | 4 |
| 六关节机械臂 | 6 | 6 | 6 |
| 两指 | 2 | 2 | 1 |
| 总计 | **19** | **18** | **11** |

夹爪同步约束减少独立运动自由度，但不会删掉 qpos/qvel 中的两个关节坐标。底座自由关节也不对应一个直接驱动器，它靠轮子与地面的接触力运动。

## 9. 检查质量、碰撞和四轮运动

本例为每个运动刚体提供带 mass 的基本 geom，由编译器推算惯量。以后使用真实数据时再填写 `<inertial>`，不要用随意的微小质量补编译错误。

保持重力、轮地接触和机械臂碰撞开启。MuJoCo 默认会过滤部分相邻刚体接触；本例没有全面禁用机器人自碰撞。若扩展后出现自碰撞卡死，逐对检查几何位置，必要时只排除明确的结构重叠对。为看清接触，可以在 viewer 的可视化选项中显示接触点和力。

建议按顺序验证：

1. 无控制运行，四轮落地，底盘不穿地。
2. 四轮同向、相同速度，观察沿 +x 前进。
3. 左右轮不同速度，观察转弯和侧滑。
4. 底盘停止，逐个控制机械臂关节。
5. 打开和关闭夹爪，确认两个指头对称运动。
6. 添加一个可移动小方块，再检查抓取接触。

后续验证底盘速度时，可给四个 drive 设置例如 1.0 的 ctrl，并在循环中调用 mj_step。当前脚本不推进物理仿真，拖动控制滑条也不会驱动机器人运动。

也可以把期望前进速度 v、角速度 omega 转成左右轮速的初始估计：

```python
r = 0.10       # 轮半径
b = 0.43       # 左右轮心距离
left = (v - omega * b / 2) / r
right = (v + omega * b / 2) / r
```

左侧前后轮使用 left，右侧前后轮使用 right。四轮滑移转向会偏离这个近似，需要根据实际接触响应调节，也要把命令限制到 ctrlrange 内。

## 10. 最后再添加门和学习任务

机器人能稳定运动之后，再独立建立门：

- 门框固定在 world 下。
- 门 body 的原点放在铰链位置，添加 z 轴 hinge；门板 geom 的中心偏到铰链一侧。
- 门把手与门板固定，或单独添加转动关节；先做无门锁版本，再添加把手与锁舌机制。
- 若训练目标是用机器人开门，门关节应由接触推动，先不用门的主动 actuator。
- 机械臂要能够接近把手，并给拉门预留底座和手臂运动空间。

先用脚本完成“到把手附近 → 对准 → 闭合 → 拉动门”，验证接触和可达性，再封装 Gymnasium。第一版 action 可用 `[v, omega, 六个关节目标, 夹爪开度]`，共 9 维，脚本将底盘两个命令映射到四轮 actuator。动作范围、控制频率、关节目标增量和力限幅都应明确；模型有 11 个 actuator 不要求策略输出也是 11 维。

当前示例只包含机器人和地面，脚本仅渲染初始姿态；尚未实现动作演示、门、逆运动学、抓取控制器或强化学习环境。

## 官方文档

- [MJCF 建模、运动学树与局部坐标](https://mujoco.readthedocs.io/en/stable/modeling.html)
- [XML 元素、形状、关节、约束与驱动器参考](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [Python 加载模型、仿真与交互 viewer](https://mujoco.readthedocs.io/en/stable/python.html)
