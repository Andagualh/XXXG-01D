#Hyperparameter Tuning
#Optimization Framework
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

class hyperparameters():
    #Tensorboard logs path
    LOG_DIR = './logs/'
    #Optuna Best Models path store
    OPT_DIR = './opt/'
    #Model save path after freq has been met
    CHECKPOINT_DIR = './train/'

    #Callback for Algorithm
    class TrainLoggingCallback(BaseCallback):
        def __init__ (self, check_freq, save_path, verbose=1):
            super(hyperparameters.TrainLoggingCallback, self).__init__(verbose)
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

    def __init__(self):
        pass
    #Optuna Range of Hyperparameters to try
    def optimize_ppo(trial):
        return {
            'n_steps':trial.suggest_int('n_steps', 2048, 8541),
            'gamma':trial.suggest_loguniform('gamma', 0.811, 0.999),
            'learning_rate':trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
            'clip_range':trial.suggest_uniform('clip_range', 0.1, 0.4),
            'gae_lambda':trial.suggest_uniform('gae_lambda', 0.811, 0.999)
        }

    def optimize_agent(trial):
        try:
            #Training Environment Creation
            env = airetro.TrainingEnv()
            env = Monitor(env, hyperparameters.LOG_DIR)
            env = DummyVecEnv([lambda:env])
            env = VecFrameStack(env, 4, channels_order='last')

            #Optuna Range of Hyperparameters to try
            model_params = hyperparameters.optimize_ppo(trial)
            #Creates a new model, comment if you want to load an existing model
            model = PPO('CnnPolicy', env, tensorboard_log=hyperparameters.LOG_DIR, verbose=0, **model_params)
            #Uses an existing model, comment if you want to use a new model
            #model = PPO.load('C:\\Users\\Acofp\\Desktop\\TFG\\OpenAIRetro\\train\\best_model_20000.zip')
            
            #Iterations
            callback = hyperparameters.TrainLoggingCallback(check_freq=10000, save_path=hyperparameters.CHECKPOINT_DIR)
            #Start training
            model.learn(total_timesteps=30000, callback=callback)

            #Model evaluation
            #n eval episode: number of games played with a set of hyperparameters
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
            env.close()

            #Save best results
            SAVE_PATH = os.path.join(hyperparameters.OPT_DIR, 'trial_{}_best_model'.format(trial.number))
            model.save(SAVE_PATH)

            return mean_reward


        except Exception as e:
            print(e)
            return -9999
 
    #Main flow of execution of this class
    def train(self):
        #Optuna Study creation
        study = optuna.create_study(direction='maximize')
        #Just use 1 job, Gym Retro doesn't supports parallel executions due to runtime limitations
        study.optimize(hyperparameters.optimize_agent, n_trials=50, n_jobs=1)


