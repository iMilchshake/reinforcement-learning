from abc import abstractmethod

import numpy as np
import pygame

from visualization import draw_circle_on_grid, visualize_numpy_array


class Environment:

    def __init__(self):
        pass

    @abstractmethod
    def get_state_count(self) -> int:
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_valid_actions(self):
        pass

    @abstractmethod
    def get_action_count(self):
        pass

    @abstractmethod
    def perform_action(self, action: int):
        pass

    @abstractmethod
    def visualize(self, *args):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        pass


class GridEnvironment(Environment):

    def __init__(self, width: int,
                 height: int,
                 move_reward: int,
                 goal_reach_reward: int,
                 invalid_move_reward: int):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=np.int32)
        self.agent_position = np.array((0, 0), dtype=np.int32)
        self.goal_position = np.array((9, 9), dtype=np.int32)
        self.move_reward = move_reward
        self.goal_reach_reward = goal_reach_reward
        self.invalid_move_reward = invalid_move_reward
        self.reset()

    def get_state_count(self):
        return self.width * self.height

    def get_state(self):
        """ map agent position to integer value representing position """
        return (self.height * self.agent_position[1]) + self.agent_position[0]

    def get_action_count(self):
        return 4

    def get_valid_actions(self):
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
        if action not in self.get_valid_actions():
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

        if self.goal_reached():
            return self.goal_reach_reward
        else:
            return self.move_reward

    def visualize(self, screen: pygame.Surface, cell_size):
        # draw grid
        visualize_numpy_array(screen, self.grid, cell_size)

        # draw player and goal
        draw_circle_on_grid(screen, self.agent_position[0], self.agent_position[1], cell_size, (0, 185, 0))
        draw_circle_on_grid(screen, self.goal_position[0], self.goal_position[1], cell_size, (150, 25, 0))

    def goal_reached(self):
        """ reward based on the current position """
        return self.agent_position[0] == self.goal_position[0] \
            and self.agent_position[1] == self.goal_position[1]

    def reset(self):
        self.agent_position[0] = 0
        self.agent_position[1] = 0

    def is_terminated(self) -> bool:
        return self.goal_reached()


if __name__ == '__main__':
    env = GridEnvironment(width=10,
                          height=10,
                          move_reward=0,
                          goal_reach_reward=1000,
                          invalid_move_reward=-100)
    print(env.get_state())
    env.agent_position[0] = 6
    print(env.get_state())
    env.agent_position[1] = 6
    print(env.get_state())
