import sys
import time
import numpy as np
import torch
from config.config import get_config
import random
from tensorboardX import SummaryWriter
import os

def mystr(x):
    x = str(x).split(',')
    res = ''
    for item in x:
        res += item + '\n \n'
    return res

class Tracker:
    def __init__(self, logdir=None):
        if logdir!=None:
            self.writer = SummaryWriter(logdir=logdir)
        else:
            config = get_config()
            if config['run_name']!='default':
                run_name = config['run_name']
            else:
                run_name = config['task_name']+'-'+config['model_name']
            self.writer = SummaryWriter(comment='-' + run_name)
            self.writer.add_text('hyper_parameter', mystr(config))
        self.path = self.writer.logdir
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []

    def train_track(self, reward, frame):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        print('%d: done %d games, reward %.3f, mean reward %.3f, speed %.2f f/s' % (
            frame, len(self.total_rewards), reward, mean_reward, speed))
        sys.stdout.flush()
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)

def test_game(env, agent, tracker, offset, test_games=1, prefix='test'):

    gamma = get_config()['gamma']
    rew_list = []
    succ = 0

    for game in range(test_games):
        rew_tmp = []
        obs = env.reset()
        step = 0
        while True:
            step += 1
            option = agent.act(obs, test=True)
            obs, reward, done, info = env.step(option)
            rew_tmp.append(reward)
            if done:
                if reward==1.0:
                    succ += 1
                sum_rew = 0
                for r in reversed(rew_tmp):
                    sum_rew = sum_rew*gamma + r
                tracker.writer.add_scalar("%s_reward"%prefix, sum_rew, offset + game)
                rew_list.append(sum_rew)
                break

    return succ/test_games, np.mean(rew_list), np.std(rew_list)

def load_dataset(path, train_ratio=0.8):
    ckp = torch.load(path)
    X = ckp['X']
    S1 = ckp['S1']
    S2 = ckp['S2']
    y = ckp['y']

    train_num = int(X.shape[0]*train_ratio)
    X_train = X[:train_num]
    S1_train = S1[:train_num]
    S2_train = S2[:train_num]
    y_train = y[:train_num]

    X_test = X[train_num:]
    S1_test = S1[train_num:]
    S2_test = S2[train_num:]
    y_test = y[train_num:]

    return X_train, S1_train, S2_train, y_train, X_test, S1_test, S2_test, y_test