from stable_baselines3.common.env_checker import check_env
from CustomPacManG import PacmanEnv

from stable_baselines3 import PPO
import os
import time

# models_dir = f"models/{int(time.time())}/"
# logdir = f"logs/{int(time.time())}/"
# if not os.path.exists(models_dir):
# 	os.makedirs(models_dir)

# if not os.path.exists(logdir):
# 	os.makedirs(logdir)
env = PacmanEnv()
#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
# # It will check your custom environment and output additional warnings if needed
# check_env(env)

episodes = 1000
max_steps = 20

for episode in range(episodes):
    step_count=0
    #print("Episode :",episode)
    #env.game_renderer._done = False
    #env.done=False
    obs = env.reset()
    while env.done==False:
        # model.learn(total_timesteps=max_steps, reset_num_timesteps=True, tb_log_name=f"PPO")
        # model.save(f"{models_dir}/{episode}")
        random_action = env.action_space.sample()
        print("action",random_action)
        obs, reward, done, info = env.step(random_action)
        #print('reward',reward)
        step_count+=1
        if step_count >= max_steps:
            env.done=True


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
