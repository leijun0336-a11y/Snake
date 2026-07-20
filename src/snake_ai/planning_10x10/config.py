from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Planner10x10Config:
    """严格 10×10 规划器配置。

    第一版有意不提供退回原始 DQN 的开关。搜索没有找到安全捷径时，
    规划器仍只会选择保持 Hamiltonian 不变量的动作。
    """

    width: int = 10
    height: int = 10
    # 完整状态 A* 只在快速候选失败时启用；500 可把 10×10 的极端帧控制在
    # 可接受范围，超过预算后仍只能采用严格验证的 Hamiltonian 路径。
    max_astar_expansions: int = 500
    # experiment8 在连续未进食步数 > 100 时终止。规划器按同一边界认证路径，
    # 即使在关闭 starvation 的历史评估协议下也不放宽安全条件。
    starvation_limit: int = 100

    def __post_init__(self) -> None:
        if (self.width, self.height) != (10, 10):
            raise ValueError("the strict planner only supports a 10x10 board")
        if self.max_astar_expansions < 1:
            raise ValueError("max_astar_expansions must be positive")
        if self.starvation_limit != self.width * self.height:
            raise ValueError("strict experiment8 planner requires starvation_limit=100")
