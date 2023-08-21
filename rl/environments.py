from abc import abstractmethod

import numpy as np
import pygame

from rl.visualization import draw_circle_on_grid, render_content, visualize_numpy_array


class EnvironmentVisualizer:
    @abstractmethod
    def visualize(self):
        pass


class Environment:

    def __init__(self):
        self.visualizer = None

    @abstractmethod
    def get_state_count(self) -> int:
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_action_count(self):
        pass

    @abstractmethod
    def perform_action(self, action: int):
        self.update_visualizer()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        pass

    def add_visualizer(self, visualizer: EnvironmentVisualizer):
        self.visualizer = visualizer

    def update_visualizer(self):
        if self.visualizer:
            self.visualizer.visualize()


class GridEnvironment(Environment):

    def __init__(self, width: int,
                 height: int,
                 move_reward: int,
                 goal_reach_reward: int,
                 wall_touch_reward: int,
                 invalid_move_reward: int,
                 random_wall_ratio: float):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = np.random.choice([0, 1],
                                     size=(width, height),
                                     p=(1 - random_wall_ratio, random_wall_ratio))
        self.agent_position = np.array((0, 0), dtype=np.int32)
        self.goal_position = np.array((width - 1, height - 1), dtype=np.int32)
        self.move_reward = move_reward
        self.goal_reach_reward = goal_reach_reward
        self.invalid_move_reward = invalid_move_reward
        self.wall_touch_reward = wall_touch_reward
        self.reset()

    def get_state_count(self):
        return self.width * self.height

    def get_state(self):
        """ map agent position to integer value representing position """
        return (self.height * self.agent_position[1]) + self.agent_position[0]

    def get_action_count(self):
        return 4

    def get_in_bounds_actions(self):
        """ 0=up, 1=right, 2=down, 3=left"""

        valid_actions = []

        if self.agent_position[1] > 0:
            valid_actions.append(0)  # up

        if self.agent_position[0] < self.width - 1:
            valid_actions.append(1)  # right

        if self.agent_position[1] < self.height - 1:
            valid_actions.append(2)  # down

        if self.agent_position[0] > 0:
            valid_actions.append(3)  # left

        return valid_actions

    def perform_action(self, action: int) -> int:
        if self.is_terminated():
            raise Exception('env is already terminated!')

        if action not in self.get_in_bounds_actions():
            return self.invalid_move_reward

        match action:
            case 0:  # up
                self.agent_position[1] -= 1
            case 1:  # right
                self.agent_position[0] += 1
            case 2:  # down
                self.agent_position[1] += 1
            case 3:  # left
                self.agent_position[0] -= 1
        self.update_visualizer()

        if self.goal_reached():
            return self.goal_reach_reward
        elif self.on_wall():
            return self.wall_touch_reward
        else:
            return self.move_reward

    def goal_reached(self) -> bool:
        return self.agent_position[0] == self.goal_position[0] \
            and self.agent_position[1] == self.goal_position[1]

    def on_wall(self) -> bool:
        return self.grid[self.agent_position[0], self.agent_position[1]] == 1

    def reset(self):
        self.agent_position[0] = 0
        self.agent_position[1] = 0

    def is_terminated(self) -> bool:
        return self.goal_reached()


class GridEnvironmentVisualizer(EnvironmentVisualizer):

    def __init__(self, cell_size: int,
                 screen: pygame.Surface,
                 env: GridEnvironment,
                 clock: pygame.time.Clock,
                 fps: int):
        self.cell_size = cell_size
        self.screen = screen
        self.env = env
        self.clock = clock
        self.fps = fps

    def visualize(self):
        # draw grid
        visualize_numpy_array(self.screen, self.env.grid, self.cell_size, wall_color=(25, 0, 0))

        # draw player and goal
        draw_circle_on_grid(self.screen, self.env.agent_position[0],
                            self.env.agent_position[1],
                            self.cell_size, (0, 185, 0))
        draw_circle_on_grid(self.screen, self.env.goal_position[0],
                            self.env.goal_position[1],
                            self.cell_size, (150, 25, 0))

        render_content(self.clock, self.fps)
