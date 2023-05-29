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
    LOG_DIR = './logs/'
    OPT_DIR = './opt/'
    CHECKPOINT_DIR = './train/'

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


    #Function to return test hyperparameters
    def optimize_ppo(trial):
        return {
            'n_steps':trial.suggest_int('n_steps', 2048, 8192),
            'gamma':trial.suggest_loguniform('gamma', 0.8, 0.999),
            'learning_rate':trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
            'clip_range':trial.suggest_uniform('clip_range', 0.1, 0.4),
            'gae_lambda':trial.suggest_uniform('gae_lambda', 0.8, 0.999)
        }

    #Training loop return -> return mean reward
    def optimize_agent(trial):
        try:
            model_params = hyperparameters.optimize_ppo(trial)

            #Enviroment summon
            env = airetro.TrainingEnv()
            env = Monitor(env, hyperparameters.LOG_DIR)
            env = DummyVecEnv([lambda:env])
            env = VecFrameStack(env, 4, channels_order='last')

            #New Algorithm
            model = PPO('CnnPolicy', env, tensorboard_log=hyperparameters.LOG_DIR, verbose=0, **model_params)
            #model = PPO.load('C:\\Users\\Acofp\\Desktop\\TFG\\OpenAIRetro\\train\\best_model_20000.zip')
            #Iterations
            callback = hyperparameters.TrainLoggingCallback(check_freq=10000, save_path=hyperparameters.CHECKPOINT_DIR)
            model.learn(total_timesteps=30000, callback=callback)

            #Model evaluation
            #n eval episode: number of games
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
            env.close()

            SAVE_PATH = os.path.join(hyperparameters.OPT_DIR, 'trial_{}_best_model'.format(trial.number))
            model.save(SAVE_PATH)

            return mean_reward


        except Exception as e:
            print(e)
            return -1000
    def train():
        #Runtime exec
        study = optuna.create_study(direction='maximize')
        #jobs = parallel execution
        study.optimize(hyperparameters.optimize_agent, n_trials=50, n_jobs=1)

        #Load Model
        #model = PPO.load('modelroute')
        #mean_reward, _ = evaluate_policy(model, env, render=True, n_eval_episodes=1)


        #Testing
        #env.step(model.predict(obs)[0])
        ## Reset game to starting state
        #obs = env.reset()
        # Set flag to flase
        #done = False
        #for game in range(1): 
        #    while not done: 
        #        if done: 
        #            obs = env.reset()
        #        env.render()
        #        action = model.predict(obs)[0]
        #        obs, reward, done, info = env.step(action)
        #        time.sleep(0.01)
        #        print(reward)


