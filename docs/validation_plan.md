# best.pt 与早停阶段验证改造计划

> 状态：已实施。本文已按当前代码更新默认流程；实现结果见
> `validation_implementation_report.md`。

## 目标

- best.pt 只由独立、无探索的验证结果决定，不再使用训练阶段的 mean_score_100。
- epsilon 降到最低点前不生成或更新正式 best.pt，也不进行早停判断。
- 早停改为根据阶段验证是否持续无提升来判断。

## 默认流程

1. 选模条件满足前正常训练，不执行选模验证；`latest.pt` 继续保存模型快照。
2. DQN 在 epsilon 首次降到最低点、PPO 在达到 `min_episodes` 时，执行 100 局快速验证和 500 局确认验证，初始化 `best.pt`。
3. 此后每 1000 个训练 episodes 执行 100 局快速验证。
4. 快速验证达到候选门槛时，再执行 500 局确认验证。
5. 只有确认验证通过时才更新 best.pt。

验证统一采用：贪心策略、无渲染、不训练、不写 replay buffer、关闭饥饿机制、固定种子、每局最多 2000 步。正式评估复用相同的策略执行核心，但拥有独立种子集和自己的默认步数上限。

## 模型选择规则

- 主指标：平均棋盘完成比例（`平均分 / 满分`），实现中换算为6×6满分33的等价
  分数尺度，以复用已有阈值。
- 次指标：满分率（6×6 满分为 33）。
- 快速验证候选门槛：6×6等价平均分提高至少 0.25；或下降不超过 0.10 且满分率提高至少 2 个百分点。
- 确认验证更新门槛：6×6等价平均分提高至少 0.15；或差距在 ±0.15 内且满分率提高至少 1.5 个百分点。
- 快速集、确认集和训练结束后的最终测试集使用互不重叠的固定种子。

## 早停规则

- `--early-stop` 默认开启，可用 `--no-early-stop` 关闭。
- DQN 在 epsilon 到达最低点且满足 `min_episodes` 后、PPO 在达到 `min_episodes` 后才允许早停。
- 连续 8 次阶段验证没有产生经过确认的新 best.pt 时，进入早停候选状态。
- 真正停止前，再对当前模型执行一次 500 局确认验证；若仍不能更新 best，才停止训练。
- target_mean_score 改为依据确认验证平均分，而不是训练 mean_score_100。

## 代码修改

1. 新增 src/snake_ai/validation.py：统一实现无探索评估、指标汇总和候选比较。
2. 修改 train.py：加入阶段验证调度、best 更新、验证 patience 和早停前确认。
3. 修改 evaluate.py：复用相同验证核心，防止训练内验证与正式评估口径漂移。
4. 新增 validation_metrics.csv，记录训练 episode、验证阶段、平均分、标准差、满分率、超时率和是否更新 best。
5. 在 best.pt 中保存入选时的训练 episode、验证指标、验证局数、种子集版本和选择规则。
6. 增加单元测试：epsilon 门控、快速/确认门槛、best 更新、patience 重置、早停前确认和固定种子复现。

## 默认新增参数

~~~text
--validation-interval 1000
--validation-episodes 100
--confirmation-episodes 500
--validation-patience 8
--validation-max-steps 2000
~~~

这些参数只控制验证，不计入 --max-episodes，也不改变 epsilon 和训练环境状态。

## 验收标准

- epsilon 未到底时不会创建或更新 best.pt，不会触发早停。
- 相同 checkpoint 和种子集重复验证结果完全一致。
- 未通过确认验证的模型不能覆盖 best.pt。
- 早停只按阶段验证轮次计算，并在停止前完成最终确认。
- `latest.pt` 始终保留最近一次模型快照，不被 `best.pt` 覆盖；它不是完整的断点续训状态。
- 完整测试和 Ruff 检查通过。

## 范围说明

本次改造不修改奖励函数、reward profile 和网络结构。发现配置冲突时应直接报错，不进行静默 fallback。
