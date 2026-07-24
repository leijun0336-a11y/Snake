# Snake AI 游戏化设计

## 1. 产品定位

将当前强化学习演示项目包装成一款简洁的 2D 贪吃蛇游戏。首版突出“玩家游玩”“观察 AI”和“双棋盘人机竞速”三种体验，同时保留训练、评估与后续玩法的扩展能力。

技术路线继续使用 Python、Pygame 和现有 PyTorch 模型，不改变 `SnakeEnv` 的训练接口。

首版统一使用 `6 × 6` 棋盘，AI 注册表中暂时只有当前最新实验 `dqn_20260715_091735` 验证选出的 `best.pt`。首版不提供 AI、难度、checkpoint 选择或模型切换界面，但底层按多 AI 扩展设计。

## 2. 设计原则

- **首版克制**：只实现完整闭环，不提前堆叠关卡、道具和养成系统。
- **逻辑与表现分离**：环境不依赖菜单、输入、动画和具体画面。
- **控制器可替换**：玩家、DQN、规则 AI 使用统一控制接口。
- **模式可插拔**：新增比赛模式时，不修改已有模式的核心逻辑。
- **配置驱动**：颜色、速度、棋盘尺寸和规则集中配置，避免散落硬编码。
- **训练兼容**：无渲染训练和现有 checkpoint 评估必须继续工作。

## 3. 首版范围（MVP）

### 3.1 功能

1. **主菜单**
   - 开始游戏
   - 观察 AI
   - 查看规则
   - 设置
   - 退出

2. **玩家模式**
   - 方向键或 WASD 控制
   - 显示分数和总 step
   - 暂停、继续、重新开始
   - 碰撞或占满棋盘后的结算页面

3. **AI 观察模式**
   - 加载唯一内置 AI
   - 调整运行速度
   - 显示分数和总 step
   - 显示当前 AI Profile ID

4. **双棋盘人机竞速**
   - 玩家与 AI 各使用一个独立的 `6 × 6` 棋盘
   - 两个棋盘使用相同规则、逻辑速度和最大 step
   - 双方使用按进食序号生成的同源食物候选序列
   - 每一侧只显示分数和总 step
   - AI 棋盘显示当前 AI Profile ID
   - 支持比赛结算和重新开始
   - 结束时先展示 `GAME OVER` 2 秒，再显示带有输赢原因的结算窗口

5. **基础设置**
   - 音量开关
   - 游戏速度：通过上下方向键或界面 `+/-` 调整，范围 1～20 tick/s，每档递增 1 tick/s，默认 6 tick/s

### 3.2 暂不实现

- 同屏双蛇对战
- 道具、技能和障碍物
- 账号、排行榜和联网功能
- 皮肤商店与成长系统
- 前端 checkpoint 选择
- 多 AI、难度档位和模型切换
- 非 `6 × 6` 棋盘

这些功能只预留扩展点，不进入首版开发范围。

## 4. 核心流程

```text
启动游戏
   ↓
主菜单 ──→ 设置
   ↓
选择玩家模式 / AI 观察模式 / 人机竞速
   ↓
游戏中 ──→ 暂停 ──→ 继续或返回菜单
   ↓
游戏结束
   ↓
重新开始 / 返回菜单
```

## 5. 界面与视觉

采用简洁的深色科技风：

| 用途 | 建议颜色 |
|---|---|
| 背景 | `#09111F` |
| 面板 | `#111D2E` |
| 玩家蛇 | `#35E6C1` |
| AI 蛇 | `#A970FF` |
| 食物 | `#FF647C` |
| 警告 | `#FFB547` |
| 主文字 | `#EEF4FF` |

首版视觉效果控制在以下范围：

- 圆角蛇身和可辨识方向的蛇头；
- 食物呼吸动画；
- 吃到食物时的轻量粒子效果；
- 蛇移动位置插值；
- 淡化网格和明确的危险边界；
- 菜单、暂停和结算使用统一面板组件。
- 主菜单和暂停菜单均可进入规则页面，返回后保持原有场景状态。

局内 HUD 不显示 Reward、饥饿度、Q 值、蛇长或 AI 动作，只显示分数和总 step。结算页面在此基础上额外显示胜负结果与结束原因。

逻辑更新维持固定 tick，渲染独立运行在 60 FPS，避免通过提高游戏速度破坏动画流畅度。

## 6. 软件结构

```text
src/snake_ai/
├── game/
│   ├── snake_env.py          # 现有单蛇规则与训练环境
│   ├── game_app.py           # 程序入口、主循环、场景切换
│   ├── session.py            # 一局游戏的生命周期
│   ├── controllers.py        # 玩家、DQN、规则 AI 控制器
│   ├── ai_profiles.py        # AI 注册表与模型配置
│   ├── food_policy.py        # 默认与竞速食物策略
│   └── modes/
│       ├── solo.py           # 玩家模式
│       ├── ai_viewer.py      # AI 观察模式
│       └── race.py           # 双棋盘人机竞速
└── ui/
    ├── theme.py              # 颜色、字号、间距
    ├── game_renderer.py      # 棋盘、实体、HUD 和粒子渲染
    ├── widgets.py            # 按钮、面板、文本等组件
    └── audio.py              # 程序生成的基础音效
```

### 6.1 分层职责

| 层 | 职责 | 不负责 |
|---|---|---|
| Environment | 移动、碰撞、食物、奖励、终止条件 | 输入、菜单、动画 |
| Controller | 根据状态产生动作 | 修改规则、直接绘图 |
| Game Mode | 组织参与者、胜负条件和局内流程 | 具体 UI 样式 |
| Scene | 菜单、暂停、结算和页面跳转 | AI 推理、碰撞计算 |
| Renderer | 将快照绘制成画面 | 改变游戏状态 |

### 6.2 稳定接口

Controller（控制器）是决策来源与游戏环境之间的适配层。它不负责移动蛇或判断碰撞，只负责把玩家按键或 AI 推理结果转换成 `SnakeEnv.step()` 接受的相对动作：`0` 直行、`1` 右转、`2` 左转。

控制器统一输出相对动作，兼容现有环境。玩家控制器额外接收 Pygame 输入事件：

```python
class Controller(Protocol):
    def reset(self) -> None: ...
    def handle_event(self, event) -> None: ...
    def choose_action(self, context: ControlContext) -> int: ...
```

- `ControlContext` 包含 observation 和当前绝对方向。
- `HumanController` 缓存最近一次有效按键，并转换为直行、右转、左转。
- `DQNController` 忽略输入事件，使用代码配置的 checkpoint 推理。
- 使用相同加载和推理方式的 DQN 只需更换 `AIProfile`，可以复用 `DQNController`。
- 如果模型类型、加载方式、推理过程或输入/动作协议不同，则新增对应 Controller；游戏模式无需随之修改。

AI 使用配置注册表管理。首版只有一个注册项，但 Controller、Game Mode 和 Scene 不得硬编码 checkpoint 路径：

```python
@dataclass(frozen=True)
class AIProfile:
    id: str
    display_name: str
    checkpoint_path: Path
    state_mode: str
    width: int
    height: int


AI_PROFILES = {
    "experiment_20260715": AIProfile(
        id="experiment_20260715",
        display_name="Snake AI",
        checkpoint_path=(
            CHECKPOINT_DIR / "dqn_20260715_091735" / "best.pt"
        ),
        state_mode="hybrid",
        width=6,
        height=6,
    ),
}

DEFAULT_AI_ID = "experiment_20260715"
```

`DQNController` 通过构造参数接收 `AIProfile`，`RaceMode` 只依赖 Controller 接口。首版从 `DEFAULT_AI_ID` 取得唯一 AI，并缓存、复用对应推理实例。启动 AI 模式时先校验 checkpoint 元数据与 Profile，不兼容或文件缺失时显示明确错误并返回菜单。

以后增加 AI 时只需新增注册项；增加选择界面时只需修改应用配置中的 `selected_ai_id`。模型加载、观察模式和竞速模式不因 AI 数量增加而修改核心流程。

渲染器只接收只读快照：

```python
@dataclass(frozen=True)
class GameSnapshot:
    snake: tuple[Point, ...]
    food: Point
    direction: Direction
    score: int
    steps: int
    hunger_ratio: float
    elapsed_seconds: float
    done: bool
    termination_reason: str
    last_action: int | None
```

环境状态更新后生成快照，UI 和动画不得反向修改环境。渲染层分别持有上一个和当前快照，并使用独立的插值进度绘制动画。

## 7. 双棋盘竞速设计

`RaceMode` 内部持有两个互不共享状态的 Session：

```text
RaceMode
├── player_session ── HumanController ── SnakeEnv(render_mode=False)
└── ai_session     ── DQNController   ── SnakeEnv(render_mode=False)
```

两个 Session 由同一个比赛时钟驱动，每个逻辑 tick 各执行一次动作。Pygame 的初始化、事件处理、渲染和关闭只由 `GameApp` 负责。

### 7.1 游戏规则

- 玩家和 AI 使用相同的 `6 × 6` 棋盘、初始蛇身和逻辑 tick。
- 游戏化模式统一设置 `starvation_enabled=False`，双方都不会因长期未进食而结束。
- Reward 只保留为 `SnakeEnv` 的内部返回值，Controller、Game Mode 和 UI 均不使用或显示。
- 训练入口及其 Reward profile、饥饿规则保持不变，不受游戏化模式影响。
- AI 始终使用 `training=False` 的确定性动作，不启用 epsilon 随机探索。
- 首个达到棋盘满分者获胜；任意一方碰撞则失败；达到最大 step 后按分数判定。
- 双方在同一个逻辑 tick 都完成动作后再统一裁决，避免执行先后顺序影响结果。

`6 × 6` 棋盘共 36 格、初始蛇长 3，因此满分为 `33`。所有 `6 × 6` 可视化单局均以 `400` 为最大 step，保留为代码配置，后续根据试玩调整。

### 7.2 同源食物候选序列

竞速采用按进食序号生成同源候选排列的方案，不能仅让两个环境使用相同 seed。双方蛇身占用格子不同，相同的可变随机调用未必产生相同食物坐标。

对第 `n` 个食物，根据比赛 seed 与食物序号生成一个确定性的全棋盘随机排列：

```text
candidate_order = permutation(
    all_36_cells,
    seed = mix(race_seed, food_index)
)

food = candidate_order 中第一个未被当前蛇占据的格子
```

- 双方第 `n` 个食物使用完全相同的候选顺序。
- 首选格对双方都合法时，食物坐标完全相同。
- 首选格被一方蛇身占据时，仅该方继续选择下一个合法候选格。
- 差异只由双方自己的路径和蛇身结构产生，不由随机调用时机产生。
- 随机排列中的第一个合法格在当前空格集合中仍保持均匀分布，尽量维持 AI 训练时的食物分布。
- 食物结果只取决于 `race_seed`、`food_index` 和当前蛇身，可以稳定回放。

食物与计分规则分别封装为可替换的 `RaceFoodPolicy` 和 `RaceScoringPolicy`。建议为 `SnakeEnv` 增加默认值不变的可选 FoodPolicy；普通训练继续使用现有随机策略，只有竞速模式注入同源策略。

### 7.3 同 tick 裁决

1. 只有一方达到满分时，该方获胜。
2. 双方同 tick 达到满分时，判为平局。
3. 只有一方碰撞时，另一方获胜。
4. 双方同 tick 碰撞时，分数高者获胜；同分则平局。
5. 达到最大 step 时，分数高者获胜；同分时，更早取得最终分数者获胜，再相同则平局。

## 8. 扩展预留

| 后续功能 | 扩展方式 |
|---|---|
| 同屏双蛇大战 | 新增 `BattleEnv` 和 `ArenaMode`，不修改 `SnakeEnv` |
| 多 AI 或新模型 | 兼容现有 DQN 时只增加 `AIProfile`；加载、推理或协议不同时新增 Controller |
| 障碍物或特殊地图 | 新增环境配置或环境实现 |
| 皮肤与主题 | 替换 Theme 和实体绘制策略 |
| 回放系统 | 记录每个 tick 的动作、随机种子和关键事件 |
| AI 分析面板 | 从 DQNController 暴露 Q 值等只读调试数据 |
| 其他棋盘尺寸 | 增加匹配尺寸的模型配置和 checkpoint |

同屏双蛇不能直接使用现有单蛇 DQN。未来应为 `BattleEnv` 增加对手状态通道并单独训练，不在首版中用临时规则污染现有环境。

## 9. 实施顺序

1. 抽离玩家输入和游戏 Session，保持现有测试通过。
2. 建立 `GameApp`、场景切换和统一 Theme。
3. 完成玩家模式、HUD、暂停和结算闭环。
4. 重制棋盘渲染并加入基础动画。
5. 封装 `DQNController`，完成 AI 观察模式。
6. 按同源食物候选序列和同 tick 裁决规则实现 `RaceMode`。
7. 增加模式、控制器和场景层的单元测试。

## 10. 首版验收标准

- 玩家可以从主菜单开始一局游戏并正常结算、重开或返回。
- AI 观察和竞速模式只使用 `dqn_20260715_091735/best.pt`，且不会写入训练数据。
- checkpoint 路径只存在于 `AI_PROFILES`，Controller、Mode 和 Scene 不硬编码具体模型。
- 人机竞速中双方由同一比赛时钟驱动，每个逻辑 tick 各推进一次。
- 玩家、AI 观察和竞速模式均不因饥饿终止，训练环境原有规则不变。
- 游戏中只显示分数和总 step，不显示训练 Reward 或 AI 内部状态。
- 竞速模式在相同比赛 seed 和双方动作序列下可以完全复现。
- 双方第 `n` 个食物使用相同的确定性候选排列，食物不会与各自蛇身重叠。
- 竞速的食物生成、计分和胜负规则不直接写死在 Scene 或 Renderer 中。
- 开启或关闭渲染不改变相同种子和动作序列的环境结果。
- 游戏速度与渲染帧率相互独立。
- `SnakeEnv` 的公开训练接口保持兼容，现有测试全部通过。
- 新增一种 Controller 或 Game Mode 时，不需要修改现有环境与渲染核心。
