import gym
import gym_sokoban

env = gym.make('Sokoban-v2')
env.reset()
env.render(mode='human')