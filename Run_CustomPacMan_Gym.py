from stable_baselines3.common.env_checker import check_env
from CustomPacManG import PacmanEnv
import os
import glob

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
import os
import time

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# model = DQN.load(f"models/1680471360/0")

env = PacmanEnv()
model = DQN(MlpPolicy, env, verbose=1, tensorboard_log=logdir)

# Set the path to the directory containing your saved models
# model_dir1 = "models/"

# Find all files in the model directory
# model_files = glob.glob(os.path.join(model_dir1, "*"))

# Find the most recently modified file
# latest_model_file = max(model_files, key=os.path.getmtime)

# Load the most recently saved model using the appropriate algorithm class
# model = DQN.load("1680811761")

episodes = 1000
max_steps = 2000

for episode in range(episodes):
    step_count = 0
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        step_count += 1
        model.learn(total_timesteps=max_steps, reset_num_timesteps=False, tb_log_name="DQN")

        if step_count >= max_steps:
            done = True

    model.save(f"{models_dir}/{episode}")



# from stable_baselines3 import PPO
# import os
# import time



# models_dir = f"models/{int(time.time())}/"
# logdir = f"logs/{int(time.time())}/"

# if not os.path.exists(models_dir):
# 	os.makedirs(models_dir)

# if not os.path.exists(logdir):
# 	os.makedirs(logdir)

# env = PacmanEnv()
# env.reset()

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# TIMESTEPS = 10000
# iters = 0

# # Training
# while True:
# 	iters += 1
# 	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
# 	model.save(f"{models_dir}/{TIMESTEPS*iters}")
 

# Testing 
# model = PPO.load("C:\Users\Gurdaan Walia\Desktop\ClassAssignment\Semester-2\CapstoneProject\models\1676930059")

# # Enjoy trained agent
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
