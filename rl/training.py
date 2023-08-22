from abc import abstractmethod
from typing import Callable

import numpy as np
import pygame
from tqdm import trange

from environments import Environment, GridEnvironment, GridEnvironmentVisualizer
from visualization import handle_quit
# from nptyping import NDarray
from numpy.typing import NDArray

RESOLUTION = (1280, 720)


class ActionSelectionAlgorithm:

    @abstractmethod
    def select_action(self, Q: np.ndarray, state: int) -> int:
        pass


class GreedyActionSelection(ActionSelectionAlgorithm):

    def select_action(self, Q: np.ndarray, state: int) -> int:
        return int(np.argmax(Q[state]))


class EpsilonGreedyActionSelection(ActionSelectionAlgorithm):

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select_action(self, Q: np.ndarray, state: int) -> int:
        if np.random.random() > self.epsilon:
            return int(np.argmax(Q[state]))
        else:
            return np.random.choice(range(Q.shape[1]))


class SoftmaxActionSelection(ActionSelectionAlgorithm):

    def __init__(self, temperature: float):
        self.temperature = temperature

    def select_action(self, Q: np.ndarray, state: int) -> int:

        raise Exception("implementation is broken")
        total = np.sum(np.exp(Q[state] / self.temperature))
        probabilities = np.exp(Q[state] / self.temperature) / total
        action = np.random.choice(Q.shape[1], p=probabilities)
        return action


class PolicyIterationAlgorithm:

    def __init__(self, env: Environment):
        self.Q = np.zeros(
            (env.get_state_count(), env.get_action_count()), dtype=np.float32)

    @abstractmethod
    def step(self):
        pass

    def run(self, iterations: int) -> NDArray[np.float32]:
        """ run algorithm for certain number of steps """
        for _ in trange(iterations):
            self.step()
        return self.Q


class SarsaPolicyIteration(PolicyIterationAlgorithm):

    def __init__(self,
                 env: Environment,
                 alpha: float,
                 gamma: float,
                 action_selection: ActionSelectionAlgorithm,
                 max_episode_length: int):

        super().__init__(env)
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discounting
        self.action_selection = action_selection
        self.max_episode_length = max_episode_length

    def step(self):
        self.env.reset()

        state = self.env.get_state()
        action = self.action_selection.select_action(self.Q, state)
        reward = self.env.perform_action(action)

        for _ in range(self.max_episode_length):
            next_state = self.env.get_state()
            next_action = self.action_selection.select_action(
                self.Q, next_state)

            # sarsa update rule
            self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action]
                                                   - self.Q[state, action])

            if self.env.is_terminated():
                break

            state = next_state
            action = next_action
            reward = self.env.perform_action(action)


class Agent:

    def __init__(self, Q: NDArray[np.float32],
                 action_selection: ActionSelectionAlgorithm):
        self.Q = Q
        self.action_selection = action_selection

    def perform_action(self, env: Environment):
        env.perform_action(
            self.action_selection.select_action(self.Q, env.get_state()))


if __name__ == '__main__':
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption("RL Visualization")
    clock = pygame.time.Clock()

    # define environment, visualizer and learning method
    env = GridEnvironment(width=10,
                          height=10,
                          move_reward=-1,
                          goal_reach_reward=1000,
                          invalid_move_reward=-100,
                          wall_touch_reward=-100,
                          random_wall_ratio=0.25)

    visualizer = GridEnvironmentVisualizer(cell_size=25,
                                           screen=screen,
                                           fps=15,
                                           clock=clock,
                                           env=env)

    sarsa = SarsaPolicyIteration(env,
                                 alpha=0.1,
                                 gamma=0.99,
                                 action_selection=EpsilonGreedyActionSelection(
                                     0.1),
                                 max_episode_length=60)

    # perform training
    visualizer.visualize()
    for _ in trange(30000):
        sarsa.step()

    # visualize fully greedy agent
    agent = Agent(sarsa.Q, GreedyActionSelection())
    env.reset()
    env.add_visualizer(visualizer)

    for _ in trange(1000):
        if env.is_terminated():
            env.reset()
        agent.perform_action(env)
        handle_quit()


# TODOS
# - decaying
