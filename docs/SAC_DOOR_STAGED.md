# 原始关节动作的两阶段 SAC 开门训练

入口：`tests/train_sac_staged.py`。环境：`src/rlinit/algorithmhand/door_sac_learn_env.py`。
底层训练实现为 `tests/train_sac_learn.py`；原 `tests/train_sac_open.py` 保留。

## 已完成的从零训练与验证

完整入口已经从随机网络正常训练至 **500000 步**，真实进程退出码为 0。
输出：`runs/door_sac_staged_v1`。最终按完整开门表现选中本次训练的 **480000 步**网络。
模型：`runs/door_sac_staged_v1/best/best_model.zip`。
SHA-256：`749cce2111ce2ef347ec47956fe75f6a7ed79316f7aaa7f66c7585d5d644a9e0`。

| 验证集合 | 结果 | 用途 |
| --- | --- | --- |
| 训练固定选模集，480000 步 | 5/5 | 选择检查点，不能代表独立成功率 |
| 入口结束时的 5 回合测试，seed 20000..20004 | 4/5 | 自动快速检查；1 次穿透失败 |
| 冻结最佳模型后的新 100 回合，seed 90000..90099 | **95/100** | 主要独立验证，全部由原带噪声 home 出发 |

这 100 回合逐步复核真实接触、下压和连续开门抓握保持时间，判定全部与环境成功标记一致。
5 次未成功中，3 次未保持开门抓握至要求时长，2 次触发原有 5 mm 穿透失败条件；没有放宽失败或成功标准。
因此已经证明在当前门模型及 home 初态扰动下可以学会任务，但没有保证每次或任意训练种子均成功。

复核轨迹：`runs/door_sac_staged_v1/verification/heldout_seed90000_100episodes.json`。
完成审计：`runs/door_sac_staged_v1/completion_audit.json`，13 项检查全部通过，确认来源、完整步数和最终交付文件哈希。
真实渲染：`runs/door_sac_staged_v1/verification/visual_seed90000/sequence.png`。
该单回合在第 217 步稳定下压至 45°，第 325 步成功，门角约 19.67°；它是动作展示，成功率来自完整 100 回合。

研究候选此前在另一组新种子 80000..80099 测得 98/100，400000 步候选为 87/100。
本次完整入口没有导入这些候选：随机训练的 100000 步前缀、自己的回放及后续网络状态均已直接比对，
480000 步时网络和优化器状态与成功研究路线逐项一致。不同种子集合的成功率存在波动，不能将旧的 98/100 写成此次新测试结果。

## 从零训练

在仓库根目录执行，输出目录必须为空：

```powershell
uv run --no-sync python tests/train_sac_staged.py --device cuda --opening-device cpu --output-dir runs/door_sac_staged
```

默认参数：seed=67、4 环境、总计 500000 步、100000 步后切换 press 奖励。
第一阶段从随机权重、原带噪声 home 姿态开始，未使用历史成功模型、示范或状态银行。
第二阶段继承本次第一阶段的 SAC 网络、优化器状态和均匀在线回放，只重算该回放中的 press 奖励。
CUDA/CPU 组合用于复现实际取得进展的实验；SAC 本身不要求切换设备。
CPU-only 可以显式指定 `--device cpu --opening-device cpu`，但不保证逐位复现 CUDA 前缀。

## 奖励切换

第一阶段保持原始接近、对齐、抓握和开门信号，抓稳后按下压程度持续奖励，最高每步约 3 分。
第二阶段移除这项持续下压加分，改成折扣势能差 `gamma * Phi(next) - Phi(current)`。
稳定抓握下压进度构成 Phi，首次达到 45° 后记录已完成下压，避免靠反复压放把手刷奖励。
真实终止时下一势能为零；超时仍保留 bootstrap。其他奖励权重与物理模型保持一致。

使用第一阶段同一次周期回调保存的 `latest_model.zip` / `latest_replay_buffer.pkl` 配对。
SB3 的回调在当前向量步入回放及梯度更新前执行：默认边界模型计数 100000、更新 94996 次、回放 99996 条。
入口检查这三个边界，拒绝过期或不同步的文件；没有把多更新 4 次的 final_model 偷换成相同配方。
从零复现实验已证明上述周期配对的网络、优化器状态及六组有效回放数组与研究分支的祖先完全一致。

## 动作和成功条件

- 保留原来的 7 个关节增量加夹爪，共 8 维动作，关节增量尺度仍为 0.01。
- 使用原模型的原生执行器；没有新增 IK、雅可比映射、规划、自动抓取或开门轨迹。
- 门把手物理范围为下压 0..60°，禁止上拉；奖励下压目标为 45°。
- 完整成功要求：从原 home 出发，先双侧稳定抓握至少 0.10 秒并下压至 45°；随后门角至少 0.30 rad（约 17.2°），并连续抓握保持 0.30 秒，无物理失败。
- 成功条件没有为了提高统计数字而放宽；比原环境额外要求真正完成稳定下压。

## 检查模型

对本次完整入口产出的最终模型做评估或可视化：

```powershell
uv run --no-sync python tests/train_sac_staged.py --evaluate runs/door_sac_staged_v1/best/best_model.zip --episodes 100 --seed 90000
uv run --no-sync python tests/train_sac_staged.py --evaluate runs/door_sac_staged_v1/best/best_model.zip --episodes 5 --seed 90000 --render
```

如需独立逐步物理复核并保存完整轨迹：

```powershell
uv run --no-sync python tests/inspect_sac_learn.py runs/door_sac_staged_v1/best/best_model.zip --episodes 100 --seed 90000 --output runs/door_sac_final_check.json
```

训练输出保留每阶段配置、XML/源文件哈希、回放和评估历史；根目录 `pipeline.json` 记录接续来源和所选模型哈希。
选模按完整开门成功率优先，避免后期 reward 或策略退化覆盖已成功的模型。
此次完整入口的 `best/best_model.zip` 已确认与冻结后测试的 480000 步文件逐字节一致。
建议使用最佳模型；训练后期固定评估曾在 460000 步降至 0/5，最后一轮训练内 5/5 也不代表任意初态都成功。
