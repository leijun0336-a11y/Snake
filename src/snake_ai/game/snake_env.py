# 处理自引用问题
from __future__ import annotations  

import random
from dataclasses import dataclass
# Enumeration（枚举） 的缩写，Python 用来定义一组固定常量的工具
from enum import Enum  

@dataclass(frozen=True)  
class Point:
    x: int
    y: int


# 对方向进行编码
class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


class SnakeEnv:
    """适合低维 DQN 训练的轻量级(类 Gym)贪吃蛇环境"""

    # 动作维度
    action_size = 3
    # 状态维度：11 个原始方向/危险特征 + 8 个距离特征。
    state_size = 19
    # 网格状态的通道数：边界、蛇身、蛇头、食物、蛇身顺序。
    grid_channels = 5

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        # 是否开启渲染
        render_mode: bool = False,
        # 每个格子的像素个数
        cell_size: int = 24,
        fps: int = 30,
        seed: int | None = None,
    ) -> None:
        if width < 5 or height < 5:
            raise ValueError("width and height must both be at least 5")

        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.cell_size = cell_size
        self.fps = fps
        self.random = random.Random(seed)
        self.renderer = None

        # 如果开启渲染，导入渲染文件包
        if render_mode:
            from snake_ai.game.renderer import SnakeRenderer

            self.renderer = SnakeRenderer(width, height, cell_size, fps)

        # 初始朝右
        self.direction = Direction.RIGHT
        # 用列表表示蛇本身，蛇用一个坐标序列来表示
        self.snake: list[Point] = []
        # 食物的初始坐标
        self.food = Point(0, 0)
        self.score = 0
        # 从上一次吃到食物之后，蛇已经走了多少步
        self.steps_since_food = 0
        # 当前 episode 已经进行了多少个环境 step，也就是这一局蛇总共走了多少步
        self.frame_iteration = 0
        self.reset()

    def reset(self) -> list[float]:
        center = Point(self.width // 2, self.height // 2)
        self.direction = Direction.RIGHT
        
        # 蛇初始在地图中间，笔直朝右，长度为3
        self.snake = [
            center,
            Point(center.x - 1, center.y),
            Point(center.x - 2, center.y),
        ]
        self.score = 0
        self.steps_since_food = 0
        self.frame_iteration = 0
        self._place_food()
        return self.get_state()

    # 让环境根据一个动作向前推进一步: 给蛇一个动作 -> 蛇走一格 -> 环境返回这一步的结果
    def step(self, action: int) -> tuple[list[float], float, bool, dict[str, int]]:
        if action not in (0, 1, 2):
            raise ValueError("action must be 0 (straight), 1 (right), or 2 (left)")

        # 蛇的总步数+1
        self.frame_iteration += 1
        # 距离上次吃到食物的步数+1
        self.steps_since_food += 1
        new_head = self._move(action)

        reward = 0.0
        done = False

        # 惩罚条件：如果这一步不会吃到食物，当前尾巴会移动走，因此允许蛇头走到尾巴原来的格子。
        if self._is_collision_after_move(new_head) or self._is_too_long_without_food():
            done = True
            reward = -10.0
            return self.get_state(), reward, done, self._get_info()

        # 奖励条件
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            reward = 10.0
            self._place_food()
        else:
            self.snake.pop()

        # 渲染
        if self.renderer is not None:
            self.renderer.render(self.snake, self.food, self.score)

        # 返回环境反馈的状态向量，奖励，是否结束，其他额外信息info.
        return self.get_state(), reward, done, self._get_info()

    # 把当前游戏局面转换成DQN能输入的状态向量。
    def get_state(self) -> list[float]:
        # 蛇头位置，是判断危险和食物方向的参考点。
        head = self.snake[0]
        # 蛇当前移动的绝对方向。
        direction = self.direction

        # 按当前方向继续直行时，下一格蛇头会到达的位置坐标。
        straight = self._next_point(direction)
        # 如果相对当前方向右转，下一格蛇头会到达的位置坐标。
        right = self._next_point(self._turn(direction, 1))
        # 如果相对当前方向左转，下一格蛇头会到达的位置坐标。
        left = self._next_point(self._turn(direction, -1))
        straight_direction = direction
        right_direction = self._turn(direction, 1)
        left_direction = self._turn(direction, -1)

        return [
            # 1. 直行方向是否危险：下一格是否会撞墙或撞到自己。
            int(self._is_collision_after_move(straight)),
            # 2. 右转方向是否危险。
            int(self._is_collision_after_move(right)),
            # 3. 左转方向是否危险。
            int(self._is_collision_after_move(left)),
            # 4. 当前移动方向是否为地图左方。
            int(direction == Direction.LEFT),
            # 5. 当前移动方向是否为地图右方。
            int(direction == Direction.RIGHT),
            # 6. 当前移动方向是否为地图上方。
            int(direction == Direction.UP),
            # 7. 当前移动方向是否为地图下方。
            int(direction == Direction.DOWN),
            # 8. 食物是否在蛇头左侧，也就是食物 x 坐标是否更小。
            int(self.food.x < head.x),
            # 9. 食物是否在蛇头右侧，也就是食物 x 坐标是否更大。
            int(self.food.x > head.x),
            # 10. 食物是否在蛇头上方，也就是食物 y 坐标是否更小。
            int(self.food.y < head.y),
            # 11. 食物是否在蛇头下方，也就是食物 y 坐标是否更大。
            int(self.food.y > head.y),
            # 12. 食物相对蛇头的 x 距离，归一化到 [-1, 1]。
            (self.food.x - head.x) / max(self.width - 1, 1),
            # 13. 食物相对蛇头的 y 距离，归一化到 [-1, 1]。
            (self.food.y - head.y) / max(self.height - 1, 1),
            # 14. 直行方向到墙的距离，值越大表示前方空间越宽。
            self._wall_distance_norm(straight_direction),
            # 15. 右转方向到墙的距离。
            self._wall_distance_norm(right_direction),
            # 16. 左转方向到墙的距离。
            self._wall_distance_norm(left_direction),
            # 17. 直行方向最近身体距离；没有身体时为 1.0。
            self._body_distance_norm(straight_direction),
            # 18. 右转方向最近身体距离；没有身体时为 1.0。
            self._body_distance_norm(right_direction),
            # 19. 左转方向最近身体距离；没有身体时为 1.0。
            self._body_distance_norm(left_direction),
        ]

    @property
    def grid_state_shape(self) -> tuple[int, int, int]:
        return self.grid_channels, self.height, self.width

    def get_grid_state(self) -> list[list[list[float]]]:
        # 通道顺序固定为：边界、蛇身、蛇头、食物、蛇身顺序。
        grid = [
            [[0.0 for _ in range(self.width)] for _ in range(self.height)]
            for _ in range(self.grid_channels)
        ]

        # 边界格子不是墙内障碍，但能提示 CNN 接近地图边缘时风险更高。
        for x in range(self.width):
            grid[0][0][x] = 1.0
            grid[0][self.height - 1][x] = 1.0
        for y in range(self.height):
            grid[0][y][0] = 1.0
            grid[0][y][self.width - 1] = 1.0

        snake_length = max(len(self.snake), 1)
        for index, point in enumerate(self.snake):
            # 蛇头为 1.0，越接近尾巴数值越小，帮助 CNN 感知身体拓扑顺序。
            order_value = (snake_length - index) / snake_length
            grid[4][point.y][point.x] = order_value
            if index == 0:
                grid[2][point.y][point.x] = 1.0
            else:
                grid[1][point.y][point.x] = 1.0

        grid[3][self.food.y][self.food.x] = 1.0
        return grid

    def get_hybrid_state(self) -> tuple[list[list[list[float]]], list[float]]:
        # Hybrid 模式同时提供完整网格和 19 维人工特征，在 Q 网络展平后拼接。
        return self.get_grid_state(), self.get_state()

    # 判断是否撞墙或者吃到蛇自己。这里是严格判断，会把当前尾巴也算作身体。
    def is_collision(self, point: Point) -> bool:
        hits_wall = point.x < 0 or point.x >= self.width or point.y < 0 or point.y >= self.height
        hits_body = point in self.snake[1:]
        return hits_wall or hits_body

    # 判断下一步是否会碰撞；没吃食物时尾巴会移动，所以不把当前尾巴算作障碍。
    def _is_collision_after_move(self, point: Point) -> bool:
        hits_wall = point.x < 0 or point.x >= self.width or point.y < 0 or point.y >= self.height
        will_eat = point == self.food
        body_to_check = self.snake[1:] if will_eat else self.snake[1:-1]
        hits_body = point in body_to_check
        return hits_wall or hits_body

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()

    def _get_info(self) -> dict[str, int]:
        return {
            "score": self.score,
            "steps": self.frame_iteration,
            "snake_length": len(self.snake),
            "steps_since_food": self.steps_since_food,
        }

    def _place_food(self) -> None:
        available = [
            Point(x, y)
            for x in range(self.width)
            for y in range(self.height)
            if Point(x, y) not in self.snake
        ]
        # 极端情况: 如果整个地图都没有地方能放食物，则直接放在蛇头的位置
        if not available:
            self.food = self.snake[0]
            return
        
        # 随机化，得到食物的坐标
        self.food = self.random.choice(available)

    # 根据动作更新蛇的朝向，并计算蛇下一步要到达的新蛇头位置
    def _move(self, action: int) -> Point:
        # 如果动作是 1，表示右转。
        if action == 1:
            self.direction = self._turn(self.direction, 1)
        # 如果动作是 2，表示左转。
        elif action == 2:
            self.direction = self._turn(self.direction, -1)

        return self._next_point(self.direction)

    # 计算下一步蛇头的坐标，移动身体在step()方法
    def _next_point(self, direction: Direction) -> Point:
        head = self.snake[0]
        # 讨论相对于蛇头的方向变动
        if direction == Direction.RIGHT:
            return Point(head.x + 1, head.y)
        if direction == Direction.LEFT:
            return Point(head.x - 1, head.y)
        if direction == Direction.DOWN:
            return Point(head.x, head.y + 1)
        return Point(head.x, head.y - 1)

    # 计算给定方向对应的坐标增量。
    @staticmethod
    def _direction_delta(direction: Direction) -> tuple[int, int]:
        if direction == Direction.RIGHT:
            return 1, 0
        if direction == Direction.LEFT:
            return -1, 0
        if direction == Direction.DOWN:
            return 0, 1
        return 0, -1

    # 计算从蛇头沿给定方向到墙之前还有多少空格，并按该轴最大距离归一化。
    def _wall_distance_norm(self, direction: Direction) -> float:
        head = self.snake[0]
        if direction == Direction.RIGHT:
            return (self.width - 1 - head.x) / max(self.width - 1, 1)
        if direction == Direction.LEFT:
            return head.x / max(self.width - 1, 1)
        if direction == Direction.DOWN:
            return (self.height - 1 - head.y) / max(self.height - 1, 1)
        return head.y / max(self.height - 1, 1)

    # 计算从蛇头沿给定方向到最近身体的距离；如果该方向没有身体，返回 1.0。
    def _body_distance_norm(self, direction: Direction) -> float:
        head = self.snake[0]
        dx, dy = self._direction_delta(direction)
        max_distance = (
            max(self.width - 1, 1)
            if direction in (Direction.RIGHT, Direction.LEFT)
            else max(self.height - 1, 1)
        )
        body = set(self.snake[1:])

        for distance in range(1, max_distance + 1):
            point = Point(head.x + dx * distance, head.y + dy * distance)
            if point.x < 0 or point.x >= self.width or point.y < 0 or point.y >= self.height:
                break
            if point in body:
                return distance / max_distance
        return 1.0

    # 调用静态方法：不需要创建实例，直接用 类名.方法名() 就能用
    # 因为它不需要访问当前环境对象 self 里的任何状态
    # 这个方法根据当前方向和转向，计算新的绝对方向
    @staticmethod
    def _turn(direction: Direction, turn: int) -> Direction:
        # direction表示当前的绝对方向，范围0到3; turn表示左转(编码为-1)或右转(编码为1)
        
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # 找到当前方向在列表里的位置，例如direction = Direction.RIGHT，则index = 0
        index = directions.index(direction)
        # 当前约定：turn = 1   表示右转；turn = -1  表示左转
        # index + 1，就是顺时针转一次，也就是右转。index - 1就是逆时针转一次，也就是左转。
        return directions[(index + turn) % len(directions)]

    # 用于判断蛇是不是太久没有吃到食物了
    def _is_too_long_without_food(self) -> bool:
        return self.steps_since_food > self.width * self.height
