"""交互观察 base.xml 的位置控制；运行方式见 --help。"""

import argparse
import math
import queue
import threading
import time
from pathlib import Path

HELP = """
终端输入：
  x y theta  设置世界坐标系下的目标位置（米、米、弧度），如 1 0 1.57
  hold       将目标设为当前位置，停止继续追踪原目标
  reset      将状态和目标复位到原点
  status     打印当前位姿和目标
  help       显示帮助
  quit       退出（也可关闭窗口或按 Ctrl+C）
窗口键盘（先点击渲染画面）：
  W/S：目标 X ±0.1 m；A/D：目标 Y ±0.1 m
  Q/E：目标转角 ±0.1 rad；H：保持当前位置；R：复位
这些是位置目标，不是速度或电机力矩指令。
"""


def read_commands(commands):
  """输入线程只提交命令；MuJoCo 状态由主线程统一修改。"""
  while True:
    try:
      line = input('目标 x y theta > ').strip()
    except EOFError:
      return
    if line:
      commands.put(line)
    if line.lower() == 'quit':
      return


def handle_command(command, model, data, target, actuator_ids, qpos_ids):
  """应用一个命令；返回 False 表示退出。"""
  import mujoco

  command = command.lower()
  increments = {
    'w': (0, 0.1),
    's': (0, -0.1),
    'a': (1, 0.1),
    'd': (1, -0.1),
    'q': (2, 0.1),
    'e': (2, -0.1),
  }
  if command in increments:
    axis, amount = increments[command]
    target[axis] += amount
  elif command in ('hold', 'h'):
    target[:] = data.qpos[qpos_ids]
  elif command in ('reset', 'r'):
    mujoco.mj_resetData(model, data)
    target[:] = 0
    mujoco.mj_forward(model, data)
  elif command == 'quit':
    return False
  elif command == 'help':
    print(HELP)
    return True
  elif command != 'status':
    try:
      values = [float(value) for value in command.split()]
      if len(values) != 3 or not all(map(math.isfinite, values)):
        raise ValueError
    except ValueError:
      print('请输入三个有限数值：x y theta，或输入 help 查看命令。')
      return True
    target[:] = values
  data.ctrl[actuator_ids] = target
  print(f'当前位姿: {data.qpos[qpos_ids].round(4)}；目标: {target.round(4)}')
  return True


def main():
  parser = argparse.ArgumentParser(
    description='渲染 base.xml 并交互设置底盘位置目标。',
    epilog=HELP,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    '--target',
    type=float,
    nargs=3,
    default=[0, 0, 0],
    metavar=('X', 'Y', 'THETA'),
    help='初始控制目标，单位为米、米、弧度。',
  )
  parser.add_argument(
    '--headless', action='store_true', help='仅运行物理仿真，不打开窗口。'
  )
  parser.add_argument(
    '--duration', type=float, help='运行指定秒数后退出（按墙钟时间）。'
  )
  args = parser.parse_args()
  if not all(map(math.isfinite, args.target)):
    parser.error('--target 必须是有限数值')
  if args.duration is not None and (
    not math.isfinite(args.duration) or args.duration <= 0
  ):
    parser.error('--duration 必须是正的有限数值')

  try:
    import mujoco
    import mujoco.viewer
    import numpy as np
  except ImportError as exc:
    parser.exit(
      1, f'缺少依赖：{exc}\n请安装：python -m pip install "mujoco>=3.1.0"\n'
    )

  # 场景仅添加地面和灯光，实际底盘模型通过 include 来自 base.xml。
  model = mujoco.MjModel.from_xml_path(
    str(Path(__file__).resolve().with_name('scene_base.xml'))
  )
  data = mujoco.MjData(model)
  names = ('joint_x', 'joint_y', 'joint_th')
  actuator_ids = [model.actuator(name).id for name in names]
  qpos_ids = [model.joint(name).qposadr[0] for name in names]
  target = np.array(args.target, dtype=float)
  data.ctrl[actuator_ids] = target
  mujoco.mj_forward(model, data)
  commands = queue.Queue()

  def key_callback(key):
    if 0 <= key < 128 and chr(key).lower() in 'wsadqehr':
      commands.put(chr(key).lower())

  viewer = None
  try:
    if not args.headless:
      viewer = mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
      )
      with viewer.lock():
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.2]
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -30
        # 只显示外观，隐藏重叠的碰撞网格。
        viewer.opt.geomgroup[3] = 0
    print(HELP)
    print(f'初始目标: {target}')
    threading.Thread(
      target=read_commands, args=(commands,), daemon=True
    ).start()
    started = time.monotonic()
    next_step = started
    next_frame = started
    running = True
    while running and (viewer is None or viewer.is_running()):
      now = time.monotonic()
      if args.duration is not None and now - started >= args.duration:
        break
      while not commands.empty():
        running = handle_command(
          commands.get_nowait(), model, data, target, actuator_ids, qpos_ids
        )
        if not running:
          break
      if not running:
        break
      # base.xml 的 position 执行器接收目标位置；保留模型中的 kp/kv。
      data.ctrl[actuator_ids] = target
      mujoco.mj_step(model, data)
      if viewer is not None and now >= next_frame:
        viewer.sync()
        next_frame = now + 1 / 60
      next_step += model.opt.timestep
      # 以接近实时的速度推进；落后时不积累无限的追赶步数。
      delay = next_step - time.monotonic()
      if delay > 0:
        time.sleep(delay)
      else:
        next_step = time.monotonic()
  except KeyboardInterrupt:
    pass
  finally:
    if viewer is not None:
      viewer.close()
    print(f'最终位姿: {data.qpos[qpos_ids].round(4)}；目标: {target.round(4)}')


if __name__ == '__main__':
  main()
