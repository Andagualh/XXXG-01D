from lib2to3.pytree import Base
import os
import optuna
#PPO Algorithm implementation
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

#Tensorboard log path
LOG_DIR = './logs/'
#Model save path
CHECKPOINT_DIR = './train/'

#Callback for Algorithm
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

class train():
    def __init__():
        callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
        #Hyperparameters
        model_params = {
            'n_steps': 8043, 'gamma': 0.8393671912093819, 'learning_rate': 1.6172427920845036e-05, 'clip_range': 0.23997123325665873, 'gae_lambda': 0.9566274038566734
        }

        #Training Environment Creation
        env = airetro.TrainingEnv()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda:env])
        env = VecFrameStack(env, 4, channels_order='last')

        folder = Path(__file__)
        #Path to load an existing model
        pathload = (folder.parent / 'models/rewardnodifferenceModelTrainedBattle2.zip').resolve()
        #Path to save a trained model, overwrites models with the same name
        pathsave = (folder.parent / 'models/rewardnodifferenceModelTrainedBattle2Test.zip').resolve()
        #Algorithm load
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR,verbose=1, **model_params)
        #Loads the model to train
        model.load(pathload)
        #Do Training
        model.learn(total_timesteps=1000000, callback=callback)
        #Saves the trained model
        model.save(pathsave)
        print('Finished training')