from abc import abstractmethod
from typing import Callable

import numpy as np
import pygame
from tqdm import trange

from environments import Environment, GridEnvironment
from visualization import display_fps, handle_quit, render_content

RESOLUTION = (1280, 720)
FPS = 0


class ActionSelectionAlgorithms:
    @staticmethod
    def epsilon_greedy(epsilon: float) -> Callable[[np.ndarray, int], int]:
        def calc(Q: np.ndarray, state: int) -> int:
            if np.random.random() > epsilon:
                return int(np.argmax(Q[state]))
            else:
                return np.random.choice(range(Q.shape[1]))

        return calc

    @staticmethod
    def greedy() -> Callable[[np.ndarray, int], int]:
        def calc(Q: np.ndarray, state: int) -> int:
            return int(np.argmax(Q[state]))

        return calc

    @staticmethod
    def softmax(temperature: float = 100) -> Callable[[np.ndarray, int], int]:
        def calc(Q: np.ndarray, state: int) -> int:
            total = np.sum(np.exp(Q[state] / temperature))
            probabilities = np.exp(Q[state] / temperature) / total
            action = np.random.choice(Q.shape[1], p=probabilities)
            return action

        return calc


def random_monte_carlo(env: Environment, iterations: int, episode_length: int):
    Q = np.zeros((env.get_state_count(), env.get_action_count()), dtype=np.int32)

    for iteration in range(iterations):
        print(iteration, np.max(Q[0]))
        env.reset()

        # get random episode
        rewards = np.zeros(episode_length, dtype=np.int32)
        states = np.zeros(episode_length, dtype=np.int32)
        actions = np.zeros(episode_length, dtype=np.int32)

        for pos in range(episode_length):
            state = env.get_state()
            # action = np.random.choice(range(env.get_action_count()))
            # action = epsilon_greedy(Q, state, epsilon=0.05)
            action = ActionSelectionAlgorithms.epsilon_greedy(0.1)(Q, state)
            reward = env.perform_action(action)

            rewards[pos] = reward
            states[pos] = state
            actions[pos] = action

            if env.is_terminated():
                break

        last_pos = pos

        # update state value
        for pos in reversed(range(last_pos)):
            future_rewards = sum(rewards[pos:])
            Q[states[pos], actions[pos]] += 0.1 * (future_rewards - Q[states[pos], actions[pos]])

    return Q


class PolicyIterationAlgorithm:

    def __init__(self, env: Environment):
        self.Q = np.zeros((env.get_state_count(), env.get_action_count()), dtype=np.float32)

    @abstractmethod
    def step(self):
        pass

    def run(self, iterations: int) -> np.ndarray[np.float32, np.float32]:
        """ run algorithm for certain number of steps """
        for _ in trange(iterations):
            self.step()
        return self.Q


class SarsaPolicyIteration(PolicyIterationAlgorithm):

    def __init__(self,
                 env: Environment,
                 alpha: float,
                 gamma: float,
                 action_selection: Callable[[np.ndarray, int], int],
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
        action = self.action_selection(self.Q, state)
        reward = self.env.perform_action(action)

        for pos in range(self.max_episode_length):
            next_state = self.env.get_state()
            next_action = self.action_selection(self.Q, next_state)

            # sarsa update rule
            self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, next_action]
                                                   - self.Q[state, action])

            if self.env.is_terminated():
                break

            state = next_state
            action = next_action
            reward = self.env.perform_action(action)


# def sarsa(env: Environment,
#           iterations: int,
#           max_episode_length: int,
#           action_selection: Callable[[np.ndarray, int], int]):
#     Q = np.zeros((env.get_state_count(), env.get_action_count()), dtype=np.float32)
#     epsilon = 0.05
#     alpha = 0.5  # learning rate
#     gamma = 0.99  # discounting
#
#     for iteration in range(iterations):
#         print(iteration, Q[0])
#         state = env.get_state()
#         action = action_selection(Q, state)
#         reward = env.perform_action(action)
#
#         for pos in range(max_episode_length):
#             next_state = env.get_state()
#             next_action = action_selection(Q, next_state)
#
#             # sarsa update rule
#             Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action]
#                                          - Q[state, action])
#
#             if env.is_terminated():
#                 break
#
#             state = next_state
#             action = next_action
#             reward = env.perform_action(action)
#
#         env.reset()
#
#     return Q


if __name__ == '__main__':
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption("Title")
    clock = pygame.time.Clock()
    env = GridEnvironment(width=10,
                          height=10,
                          move_reward=-1,
                          goal_reach_reward=100,
                          invalid_move_reward=-100)

    while True:
        handle_quit()
        screen.fill("white")

        env.visualize(screen, cell_size=35)
        sarsa = SarsaPolicyIteration(env,
                                     alpha=0.1,
                                     gamma=0.99,
                                     action_selection=ActionSelectionAlgorithms.epsilon_greedy(0.05),
                                     max_episode_length=50)
        optimal_Q = sarsa.run(10_000)
        optimal_V = np.reshape(np.max(optimal_Q, -1), (-1, 10))
        optimal_policy = np.reshape(np.argmax(optimal_Q, -1), (-1, 10))

        display_fps(screen, clock)
        render_content(clock, FPS)
