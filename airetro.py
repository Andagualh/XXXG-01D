import retro
import os
import time
from gym import Env
from gym.spaces import MultiBinary, Box
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class TrainingEnv(Env):
	def __init__(self):
		super().__init__()
		self.enemyhealth = 0
		self.penaltyTime = 0
		self.i = 0
		self.observation_space = Box(low =0, high = 255, shape=(82,82,1), dtype=np.uint8)
		self.action_space = MultiBinary(12)

		#retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "custom_integrations"))
		self.game = retro.make("GundamW-Snes", use_restricted_actions=retro.Actions.FILTERED)

		
	def EnemyHealthReward(self,info):
		rewardHP = info['enemyhealth'] - self.enemyhealth 
		self.enemyhealth = info['enemyhealth']
		return rewardHP
	
	def PlayerHealthPenalty(self, info):
		rewardHP = self.playerhealth - info['playerhealth']
		self.playerhealth = info['playerhealth']
		return rewardHP
	
	def PlayerComboReward(self, info):
		rewardCombo = info['hitcomboplayer'] - self.hitcomboplayer
		self.hitcomboplayer = info['hitcomboplayer']
		return rewardCombo
	
	def TimePenalty(self):
		penaltyTime = self.penaltyTime
		self.penaltyTime = self.penaltyTime + 1
		return penaltyTime
	
	def penaltyLose(self, info):
		if info['playerhealth'] == 0:
			return -100
		return 0
	
	def winBonus(self, info):
		if info['enemyhealth'] == 0:
			return 100
		return 0
	
	def step(self, action):
		#First Step
		obs, reward, done, info = self.game.step(action)
		ogobs = obs
		#self.capture(obs)
		obs = self.preprocess(obs)

		#Frame Delta Calculus
		frame_delta = obs - self.previous_frame
		self.previous_frame = obs

		if info['enemyhealth'] < self.enemyhealth:
			reward = (info['enemyhealth'] - self.enemyhealth)*-1
			self.enemyhealth = info['enemyhealth']
		else:
			reward = 0
			self.enemyhealth = info['enemyhealth']
		
		return frame_delta, reward, done, info
	
	def render(self, *args, **kwargs):
		self.game.render()
	
	def reset(self):
		#First Frame
		obs = self.game.reset()
		obs = self.preprocess(obs)

		self.previous_frame = obs
		#Score Delta
		self.enemyhealth = 0
		self.playerhealth = 0
		return obs

	def preprocess(self, observation):
		#Grayscale, Frame Delta, Resize Frame
		gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY) 
		resize = cv2.resize(gray,(82,82), interpolation = cv2.INTER_CUBIC)
		channels = np.reshape(resize, (82,82,1))
		return channels

	def winState(self, observation):
		color = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
		flag = False

		#gray = gray[99:118,51:68]
		
		gray = gray[45:68, 99:120]
		color = color[45:68,99:120]

		#cv2.imwrite(str('images/fullrun/frame') + str(self.i) + str('.jpg'), color)
		#self.i = self.i+1

		template = cv2.imread('images\samescaleWIN.jpg')
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		#w, h = template.shape[::1]
		cv2.imshow('gray',template)
		res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
		threshold = 0.8

		w, h = template.shape[::-1]

		if np.amax(res) > threshold:
			flag = True	
		
		print(flag)
		return flag
	
	def loseState(self, observation):
		color = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
		flag = False

		#gray = gray[99:118,51:68]
		
		gray = gray[45:68, 135:150]
		color = color[45:68,135:150]

		template = cv2.imread('images\samescaleWIN.jpg')
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		#w, h = template.shape[::1]
		res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
		threshold = 0.8

		w, h = template.shape[::-1]
		if np.amax(res) > threshold:
			flag = True
		
		if np.all(res) == 1:
			flag = False
		
		print(flag)
		print(res)
		return flag
	
	def capture(self, observation):
		color = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
		cv2.imwrite('images/frame.jpg', color)
	
	def close(self):
		self.game.close()

#Execution Runtime
# env = TrainingEnv()
# env.observation_space.shape
# env.action_space.shape

# obs = env.reset()
# done = False
# for game in range(100):
# 	while not done:
# 		if done:
# 			obs = env.reset()
# 		env.render()
# 		obs, reward, done, info = env.step(env.action_space.sample())
# 		time.sleep(0.01)	
# 		if reward > 0:
# 			print(reward)