import numpy as np
from config.config import get_config
import random
import os
import torch

class Replay():

    def __init__(self):
        self.cf = get_config()
        self.memory_size = self.cf['buffer_size']
        self.bs = self.cf['batch_size']
        self.options = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.float32)
        self.maps = np.empty((self.memory_size, 2, self.cf['imsize'], self.cf['imsize']), dtype = np.uint8)
        self.xys = np.empty((self.memory_size, 2), dtype = np.uint8)
        self.current = 0
        self.count = 0

        self.premap = np.empty((self.bs, 2, self.cf['imsize'], self.cf['imsize']), dtype = np.uint8)
        self.prexy = np.empty((self.bs, 2), dtype = np.uint8)
        self.reset_tmp()

    def reset_tmp(self):
        self.obs_tmp = []
        self.reward_tmp = []
        self.option_tmp = []

    def add_tmp(self, obs, reward, option):
        self.obs_tmp.append(obs)
        self.reward_tmp.append(reward)
        self.option_tmp.append(option)

    def popall(self):
        n = len(self.option_tmp)
        R = 0
        for i in reversed(range(n)):
            R = self.reward_tmp[i] + self.cf['gamma']*R
            self.add(self.obs_tmp[i], R, self.option_tmp[i])

        self.reset_tmp()

    def add(self, obs, reward, option):
        ma = np.moveaxis(obs[0], -1, 0)
        xy = obs[1]
        self.options[self.current] = option
        self.rewards[self.current] = reward
        self.maps[self.current, ...] = ma
        self.xys[self.current, ...] = xy
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self):

        indexes = []
        while len(indexes) < self.bs:
            index = random.randint(0, self.count - 1)
            self.premap[len(indexes), ...] = self.maps[index, ...]
            self.prexy[len(indexes), ...] = self.xys[index, ...]
            indexes.append(index)
        
        premap = torch.cuda.FloatTensor(self.premap)
        prexy = torch.cuda.LongTensor(self.prexy).view(-1,1,2)
        options = torch.cuda.LongTensor(self.options[indexes]).view(-1,1,1)
        rewards = torch.cuda.FloatTensor(self.rewards[indexes])

        pres1 = prexy[:,:,0].view(-1,1)
        pres2 = prexy[:,:,1].view(-1,1)

        return premap, pres1, pres2, options, rewards