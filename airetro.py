import retro
import os
import time
from gym import Env
from gym.spaces import MultiBinary, Box
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

class TrainingEnv(Env):
	def __init__(self):
		super().__init__()
		#Restart class variables
		self.enemyhealth = 0
		self.penaltyTime = 0
		self.i = 0
		#Define observation space for environment
		self.observation_space = Box(low =0, high = 255, shape=(82,82,1), dtype=np.uint8)
		self.action_space = MultiBinary(12)

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
		obs, reward, done, info = self.game.step(action)
		ogobs = obs
		obs = self.preprocess(obs)

		#Frame Delta Matrix for NN
		frame_delta = obs - self.previous_frame
		self.previous_frame = obs
		
		#Reward definition 
		if info['enemyhealth'] < self.enemyhealth:
			reward = (info['enemyhealth'] - self.enemyhealth)*-1
			self.enemyhealth = info['enemyhealth']
		else:
			reward = 0
			self.enemyhealth = info['enemyhealth']
		
		#Entry for Neuronal Network
		return frame_delta, reward, done, info
	
	#Render call
	def render(self, *args, **kwargs):
		self.game.render()
	
	#Reset call
	def reset(self):
		obs = self.game.reset()
		obs = self.preprocess(obs)

		self.previous_frame = obs
		self.enemyhealth = 0
		self.playerhealth = 0
		return obs

	def preprocess(self, observation):
		#Grayscale, Frame Delta, Resize Frame
		gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY) 
		resize = cv2.resize(gray,(82,82), interpolation = cv2.INTER_CUBIC)
		channels = np.reshape(resize, (82,82,1))
		return channels

	#Unfinished, only triggers when Full Black on Screen, flag used for Done state
	def winState(self, observation):
		color = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
		flag = False

		#Area to scan on the view [Y axis, X axis]
		gray = gray[45:68, 99:120]
		color = color[45:68,99:120]

		#Sprite to detect load
		template = cv2.imread('images\samescaleWIN.jpg')
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		cv2.imshow('gray',template)
		res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
		threshold = 0.8

		w, h = template.shape[::-1]

		if np.amax(res) > threshold:
			flag = True	
		
		print(flag)
		return flag
	#Unfinished, only triggers when Full Black on Screen, flag used for Done state
	def loseState(self, observation):
		color = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
		flag = False

		#Area to scan on the view [Y axis, X axis]
		gray = gray[45:68, 135:150]
		color = color[45:68,135:150]

		#Sprite to detect load
		template = cv2.imread('images\samescaleWIN.jpg')
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		cv2.imshow('gray',template)
		res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
		threshold = 0.8

		w, h = template.shape[::-1]

		if np.amax(res) > threshold:
			flag = True	
		
		print(flag)
		return flag
	
	#Debug method to capture frames to an jpg image
	def capture(self, observation):
		color = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
		cv2.imwrite('images/frame.jpg', color)
	
	def close(self):
		self.game.close()
