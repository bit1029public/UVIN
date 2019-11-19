import torch
from config.config import get_config
import util
from agent import Actor, Learner
import os
import numpy as np
import importlib

config = get_config()
os.environ['CUDA_VISIBLE_DEVICES'] = config['visible_device']
environment = importlib.import_module('environment.'+config['task_name'])

ds = np.zeros([config['imsize']**2, config['A_size'], config['C_size'], 2], dtype=np.int)
learner = Learner(ds)
tracker = util.Tracker()

env = environment.Env()
actor = Actor(learner.main_net)
preobs = env.reset()
rew_tmp = []

for frame_idx in range(config['max_training_step']):
    option = actor.act(preobs)
    postobs, reward, done, info = env.step(option)
    learner.main_net.add_sample(preobs[1],option,postobs[1])
    learner.buffer.add_tmp(preobs, reward, option)
    preobs = postobs
    rew_tmp.append(reward)
    if done:
        sum_rew = 0
        learner.buffer.popall()
        for r in reversed(rew_tmp):
            sum_rew = config['gamma']*sum_rew + r
        tracker.train_track(sum_rew, frame_idx)
        rew_tmp = []
        preobs = env.reset()
    if frame_idx%config['learn_step']==0 and frame_idx>config['init_memory_step']:
        loss = learner.learn()
        tracker.writer.add_scalar("loss", loss, frame_idx)
    if frame_idx%config['update_ds_step']==0:
        learner.main_net.update_ds()
    
model_path = os.path.join(tracker.path, 'model')
torch.save({'net':learner.main_net.state_dict(),
            'ds':learner.main_net.ds,
            'optimizer':learner.optimizer.state_dict()},
                model_path)
sr,mean,std = util.test_game(env, actor, tracker, 0, config['test_final_game'])
print('test sr:%f, reward: mean:%f std:%f'%(sr, mean, std))