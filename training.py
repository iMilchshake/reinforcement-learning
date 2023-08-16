# Settings

import numpy as np
import pygame

from environments import Environment, GridEnvironment
from visualization import display_fps, handle_quit, render_content

RESOLUTION = (1280, 720)
FPS = 0


def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float):
    if np.random.random() > epsilon:
        return greedy(Q, state)
    else:
        return np.random.choice(range(Q.shape[1]))


def greedy(Q: np.ndarray, state: int):
    return np.argmax(Q[state])


def softmax(Q: np.ndarray, state: int, temporance=100):
    total = np.sum(np.exp(Q[state] / temporance))
    probabilities = np.exp(Q[state] / temporance) / total
    # print(probabilities)
    index = np.random.choice(Q.shape[1], p=probabilities)

    return index


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
            action = softmax(Q, state)
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


if __name__ == '__main__':
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption("Title")
    clock = pygame.time.Clock()
    env = GridEnvironment(width=10,
                          height=10,
                          move_reward=-1,
                          goal_reach_reward=1000,
                          invalid_move_reward=-20)

    while True:
        handle_quit()
        screen.fill("white")

        # print(env.get_valid_actions())
        # print(env.agent_position)
        # reward = env.perform_action(np.random.choice(range(env.get_action_count())))
        # print(env.agent_position, reward)
        env.visualize(screen, cell_size=35)
        optimal_Q = random_monte_carlo(env, iterations=10000, episode_length=50)
        optimal_V = np.max(optimal_Q, -1)
        optimal_actions = np.argmax(optimal_Q, -1)

        B = np.reshape(optimal_V, (-1, 10))
        A = np.reshape(optimal_actions, (-1, 10))

        display_fps(screen, clock)
        render_content(clock, FPS)
