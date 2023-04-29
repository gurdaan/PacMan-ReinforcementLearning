import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from CustomPacManG import PacmanEnv

# Create a directory to store the training logs and models
logdir = './logs'
os.makedirs(logdir, exist_ok=True)

# Create a Pacman environment
env = PacmanEnv()

# Wrap the environment in a vectorized environment to support parallel environments
env = DummyVecEnv([lambda: env])

# Normalize the environment observations to improve training stability
env = VecNormalize(env)

# Create a PPO model with a multi-layer perceptron policy network
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# # Train the model on the Pacman environment for 100,000 steps
# model.learn(total_timesteps=100)
# model.save("/model")

model = PPO.load("model")
# Evaluate the model on 10 episodes
for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print(f"Episode {episode}: Total reward = {total_reward}")
