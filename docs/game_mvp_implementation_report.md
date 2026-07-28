# Snake AI 游戏化 MVP 实现报告

> 当前注册表和启动行为已按现有代码更新；测试数量记录的是 MVP 初次实现时的结果。

## 1. 完成结果

本次实现将原有训练/评估项目扩展为可直接启动的 Pygame 游戏，同时保持 `SnakeEnv` 原有训练调用兼容。

首版固定使用 `6 × 6` 棋盘，包含：

- 玩家单人模式；
- 唯一内置 AI 观察模式；
- 双棋盘人机竞速；
- 主菜单、设置、暂停、重新开始与结算；
- 可从主菜单或暂停菜单进入的游戏规则页面；
- 60 FPS 渲染、逻辑 tick 分离、移动插值、食物呼吸与进食粒子；
- 程序生成的按钮、进食和结束音效；
- 仅显示 Score 和 Steps 的局内 HUD；
- 所有可视化单局最大 400 step。

## 2. 启动与操作

项目需要 Python 3.12+、Pygame、NumPy 和可用的 PyTorch 环境。同步项目依赖后运行：

```bash
uv sync --extra cpu
uv run --extra cpu snake-play
```

也可以直接运行模块：

```bash
uv run --extra cpu python -m snake_ai.game.game_app
```

主菜单模式：

| 菜单 | 功能 |
|---|---|
| `PLAY SOLO` | 玩家单人游戏 |
| `WATCH AI` | 观察内置 AI 自动游戏 |
| `HUMAN VS AI` | 玩家与 AI 双棋盘竞速 |
| `RULES` | 查看棋盘、胜负、竞速和操作规则 |
| `SETTINGS` | 调整逻辑速度和声音 |

游戏结束后先在最终棋盘上显示 `GAME OVER` 2 秒，再进入结算窗口；窗口会显示结果、分数、step 和简短的输赢原因。

操作：

- 方向键或 `WASD`：转向；
- `P` 或 `Esc`：暂停/继续；
- 鼠标左键：选择菜单按钮；
- 设置页速度档位：1～20 tick/s，使用上下方向键或界面 `+/-` 调整，每档递增 1 tick/s，默认 6 tick/s；
- 渲染始终保持 60 FPS。

玩家单人和人机竞速模式每次开局（包括 `PLAY AGAIN`）都有 3 秒倒计时，倒计时期间接收方向输入。AI checkpoint 在主菜单首帧后后台预加载；若尚未完成则显示加载界面，避免同步加载阻塞菜单。

PyTorch 使用互斥的 `cpu` 与 `cu124` 可选依赖。`uv run --extra cpu snake-play` 会在首次运行时同步 CPU 游戏环境；Windows 用户也可以双击根目录的 `start_game.bat`。Windows AutoDL 脚本显式选择 `cu124` 并验证 CUDA；Linux 脚本只同步默认依赖并检查当前环境，不会自动选择 PyTorch extra。

## 3. 游戏规则

### 3.1 通用规则

- 棋盘固定为 `6 × 6`；
- 游戏化模式关闭饥饿终止；
- Reward 仍由环境内部计算，但游戏模式、Controller 和 UI 不读取或显示；
- 碰撞、占满棋盘或达到 400 step 时结束单局；
- 局内只显示分数和总 step。

训练入口的 Reward profile 与饥饿规则没有修改。

### 3.2 AI

当前注册表只有一个 AI：

```text
profile_id: dqn_20260722_201922
checkpoint: checkpoints/dqn_20260722_201922/best.pt
state_mode: hybrid
network: q_network
board: 6 × 6
reward_profile: experiment8
```

加载时严格校验文件、棋盘尺寸、状态模式、Reward profile、网络类型、状态尺寸和架构版本。不兼容时直接报错并返回菜单，不会改用 `latest.pt`、其他 checkpoint 或其他模型。

### 3.3 双棋盘竞速

- 双方使用相同初始状态、比赛 seed、逻辑速度和裁决时钟；
- 每个 tick 先分别计算双方动作，再推进两个环境，最后统一裁决；
- 先达到棋盘满分 33 分者获胜；
- 一方碰撞时另一方获胜；
- 第 400 tick 后按分数裁决；同分时，先取得最终分数者获胜，再相同则平局；
- 双方同 tick 达标或同分碰撞时按明确的平局/分数规则处理，不依赖代码执行顺序。

公平食物采用“同源候选排列”：

```text
local_seed = stable_hash(race_seed, food_index)
candidate_order = shuffle(all_36_cells, local_seed)
food = candidate_order 中第一个不在当前蛇身上的格子
```

双方第 `n` 个食物使用相同候选顺序。首选格都合法时坐标完全相同；被某一方蛇身占据时，仅该方取下一个合法候选。该过程可复现，并保持合法空格上的均匀分布。

## 4. 实现结构

### 4.1 环境兼容

`SnakeEnv` 只增加了一个可选 `food_policy` 参数：

- 默认 `RandomFoodPolicy` 继续调用原有随机数生成器并均匀选择空格；
- 竞速模式显式注入 `SeededRaceFoodPolicy`；
- 不传参数时，训练和评估行为保持不变。

### 4.2 Controller

Controller 只负责把决策来源转换成环境动作：

```text
HumanController ── 键盘输入 ──→ 0/1/2
DQNController   ── 模型推理 ──→ 0/1/2
                                  ↓
                           SnakeEnv.step()
```

`HumanController` 使用两步输入缓冲，拒绝直接反向并保留连续转弯输入。`DQNController` 通过 `AIProfile` 注入模型配置，游戏模式不包含 checkpoint 路径。

### 4.3 AI 扩展

`AI_PROFILES` 当前只有一个注册项，`DEFAULT_AI_ID` 指向它。以后增加兼容 DQN 时只需新增 Profile；加载或推理方式改变时新增对应 Controller。观察模式和竞速模式只依赖 Controller 接口，不需要随 AI 数量修改。

### 4.4 Session 与 Mode

`GameSession` 负责：

- reset、动作选择和单步推进；
- 400 step 上限；
- 只读 `GameSnapshot`；
- 上一帧/当前帧状态，供渲染插值使用。

三种 Mode 分别组织单人、AI 观察和双 Session 竞速。`GameApp` 统一拥有 Pygame 生命周期、事件队列、场景切换和逻辑时钟，环境始终使用 `render_mode=False`。

### 4.5 渲染与声音

- 深色科技风主题；
- 玩家青色、AI 紫色、食物红色；
- 圆角蛇身、方向眼睛、淡网格；
- 逻辑位置之间插值；
- 食物呼吸与确定性粒子；
- 音效由 NumPy 生成波形，不依赖外部音频资源。

## 5. 主要文件

```text
src/snake_ai/game/
├── ai_profiles.py
├── controllers.py
├── food_policy.py
├── game_app.py
├── session.py
└── modes/
    ├── solo.py
    ├── ai_viewer.py
    └── race.py

src/snake_ai/ui/
├── audio.py
├── game_renderer.py
├── theme.py
└── widgets.py

tests/test_gameplay.py
```

命令入口在 `pyproject.toml` 中注册为 `snake-play`。

## 6. 初次实现时的验证结果

执行：

```bash
python -m ruff check src/snake_ai/game src/snake_ai/ui tests/test_gameplay.py
python -m pytest -q -p no:cacheprovider
```

结果：

- Ruff：通过；
- Pytest：`87 passed`；
- Pygame 无窗口烟雾测试：单人、AI 观察和人机竞速均能创建、推进和渲染；
- 当时注册的真实 checkpoint：严格加载并成功输出动作。

Pytest 仅报告 Pygame 间接使用 `pkg_resources` 的弃用警告，不影响本次功能。

## 7. 当前边界

- 前端不提供 AI、checkpoint 或棋盘尺寸选择；
- 当前只有一个 `6 × 6` AI Profile；
- 不包含同屏双蛇、障碍物、特殊地图、联网和排行榜；
- 新棋盘尺寸必须配套相同尺寸的 checkpoint 后再注册；
- 音频设备初始化失败会明确抛出错误，不会静默关闭声音。
