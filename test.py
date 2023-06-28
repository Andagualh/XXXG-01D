from lib2to3.pytree import Base
import os
import optuna
#PPO Algorithm implementation (Subject to change)
from stable_baselines3 import PPO
import stable_baselines3
#Evaluation policy
from stable_baselines3.common.evaluation import evaluate_policy
#Logging
from stable_baselines3.common.monitor import Monitor
#Vec Wrappers
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import airetro
import time
from pathlib import Path

#Starts an execution using a trained model.

class test():
    #Log Path of Execution
    LOG_DIR = './logs/'
    def __init__(self):
        pass
    def start(self):
        #Training Environment Creation
        env = airetro.TrainingEnv()
        env = Monitor(env, test.LOG_DIR)
        env = DummyVecEnv([lambda:env])
        env = VecFrameStack(env, 4, channels_order='last')

        #Load Model
        folder = Path(__file__)
        #Model Loading Path
        path = (folder.parent / 'models/rewardnodifferenceModelTrainedBattle2.zip').resolve()
        model = PPO.load(path)
        obs = env.reset()

        env.step(model.predict(obs)[0])
        obs = env.reset()
        done = False
        #Change range for multiple combats
        for game in range(1): 
            while not done: 
                if done: 
                    obs = env.reset()
                #Renders the emulator output in a window
                env.render()
                action = model.predict(obs)[0]
                obs, reward, done, info = env.step(action)
                time.sleep(0.01)
                #Prints every reward that is not Zero
                if reward != 0:
                    print(reward)
