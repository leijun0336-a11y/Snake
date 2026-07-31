# 问题记录

本文记录已经定位或解决过的问题。当前行为以 `src/snake_ai/`、`README.md` 和测试为准。

## 走到即将离开的尾巴位置被误判为碰撞

**状态：已解决。**

环境会区分本步是否吃到食物：

- `new_head == food`：蛇会增长，尾巴不会移动，因此蛇头不能进入任何现有身体格；
- `new_head != food`：尾巴会在本步离开，蛇头可以进入移动前的尾巴格。

当前判断由 `SnakeEnv._is_collision_after_move()` 统一处理。

## 多次训练互相覆盖 checkpoint

**状态：已解决。**

每次训练使用带算法和时间戳的独立目录：

```text
checkpoints/<algorithm>_<timestamp>/
runs/<algorithm>_<timestamp>/
```

其中 `latest.pt` 是最近一次模型快照，`best.pt` 由阶段验证选出。当前 checkpoint
不包含 optimizer、replay/rollout buffer 和随机数状态，因此不能视为完整的断点续训文件。


