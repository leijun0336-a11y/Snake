# 阶段验证功能实现报告

## 功能变更

- best.pt 不再依据训练 mean_score_100，而是依据独立贪心验证。
- epsilon 到达下限时运行 100 局快速验证和 500 局确认验证，初始化 best.pt。
- 此后默认每 500 个训练 episode 快速验证一次；只有确认验证通过才更新 best.pt。
- 早停改为按阶段验证轮次计数，连续 8 轮无确认提升时，在停止前补做或复用确认验证。
- target_mean_score 改为使用确认验证均分。
- latest.pt 始终保存最后训练状态，不会被 best.pt 覆盖。

## 实现结构

- 新增 src/snake_ai/validation.py，集中放置验证执行、候选比较、固定种子和阶段验证状态机。
- train.py 主循环只调用状态机、记录事件并保存 checkpoint，避免继续堆叠选模分支。
- evaluate.py 复用同一贪心验证核心，并使用独立的最终测试种子集。
- SnakeEnv.reset() 新增可选 seed，使每个验证 episode 都能独立复现。

## 新增产物与参数

- 新增 runs/<run_name>/validation_metrics.csv。
- best.pt 记录入选训练 episode、快速/确认验证指标、种子集信息和门槛配置。
- 新增参数：

~~~text
--validation-interval 500
--validation-episodes 100
--confirmation-episodes 500
--validation-patience 8
--validation-max-steps 1000
~~~

旧的 --patience 和 --min-delta 已移除。

## 验证结果

- Ruff：通过。
- 完整测试：62 passed。
- 2 个训练 episode 的端到端 smoke test：成功生成 best.pt、latest.pt、config.json 和 validation_metrics.csv，并正确走通早停前确认流程。
- smoke test 临时产物已清理；没有在本地启动正式训练。

## 范围说明

本次没有修改奖励函数、reward profile、网络结构和训练超参数。epsilon 未在训练结束前到达下限时不会生成 best.pt，程序会明确提示并只保留 latest.pt，不会静默 fallback。
