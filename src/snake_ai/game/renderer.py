# 处理自引用问题
from __future__ import annotations

from snake_ai.game.snake_env import Point


class SnakeRenderer:
    SNAKE_GAP = 8
    FOOD_GAP = 10

    def __init__(self, width: int, height: int, cell_size: int = 48, fps: int = 20) -> None:
        # pygame 只在真正开启渲染时导入，避免无渲染训练时强依赖窗口环境。
        import pygame

        self.pygame = pygame
        # 游戏网格宽度，单位是格子数量。
        self.width = width
        # 游戏网格高度，单位是格子数量。
        self.height = height
        # 每个格子渲染成多少像素。
        self.cell_size = cell_size
        # 渲染帧率，控制画面刷新速度，影响蛇跑动的速度。
        self.fps = fps
        # pygame 窗口宽度，等于网格宽度 * 单格像素。
        self.screen_width = width * cell_size
        # pygame 窗口高度，下面额外加 44 像素用于显示分数栏。
        self.screen_height = height * cell_size + 44

        # 初始化 pygame。
        pygame.init()
        # 设置窗口标题。
        pygame.display.set_caption("Snake AI")
        # 创建游戏窗口。
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        # 创建时钟对象，用于控制 fps。
        self.clock = pygame.time.Clock()
        # 使用 pygame 内置默认字体，避免 SysFont 扫描 Windows 字体注册表时因异常的
        # 非字符串字体项触发 TypeError；None 表示不依赖本机安装的系统字体。
        self.font = pygame.font.Font(None, 24)

    # 根据当前蛇、食物和分数刷新一帧画面。
    def render(self, snake: list[Point], food: Point, score: int) -> None:
        pygame = self.pygame
        # 处理窗口事件；如果用户点击关闭按钮，就退出渲染。
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

        # 清空上一帧画面，填充背景色。
        self.screen.fill((20, 24, 28))
        # 绘制网格线。
        self._draw_grid()
        # 绘制食物。
        self._draw_food(food)
        # 绘制蛇。
        self._draw_snake(snake)

        # 渲染分数文本。
        score_text = self.font.render(f"Score: {score}", True, (235, 239, 244))
        # 把分数文本放到游戏区域下方的分数栏。
        self.screen.blit(score_text, (12, self.height * self.cell_size + 9))

        # 把内存中的绘制结果显示到窗口上。
        pygame.display.flip()
        # 控制渲染速度，避免窗口刷新过快。
        self.clock.tick(self.fps)

    # 关闭 pygame。
    def close(self) -> None:
        self.pygame.quit()

    # 把网格坐标转换成 pygame 使用的像素矩形。
    def _cell_rect(self, point: Point):
        return self.pygame.Rect(
            point.x * self.cell_size,
            point.y * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    # 绘制游戏区域的网格线。
    def _draw_grid(self) -> None:
        pygame = self.pygame
        color = (35, 40, 46)
        # 绘制竖线。
        for x in range(self.width + 1):
            px = x * self.cell_size
            pygame.draw.line(self.screen, color, (px, 0), (px, self.height * self.cell_size))
        # 绘制横线。
        for y in range(self.height + 1):
            py = y * self.cell_size
            pygame.draw.line(self.screen, color, (0, py), (self.screen_width, py))

    # 绘制食物，食物位置由网格坐标 food 决定。
    def _draw_food(self, food: Point) -> None:
        food_rect = self._cell_rect(food).inflate(-self.FOOD_GAP, -self.FOOD_GAP)
        self.pygame.draw.rect(self.screen, (230, 70, 70), food_rect)

    # 绘制蛇；snake[0] 是蛇头，其余是身体。
    def _draw_snake(self, snake: list[Point]) -> None:
        for index, point in enumerate(snake):
            # 蛇头颜色更亮，身体颜色稍暗。
            color = (80, 210, 120) if index == 0 else (55, 160, 95)
            snake_rect = self._cell_rect(point).inflate(-self.SNAKE_GAP, -self.SNAKE_GAP)
            self.pygame.draw.rect(self.screen, color, snake_rect)
