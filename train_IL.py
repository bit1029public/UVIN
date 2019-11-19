from config.config import get_config
import numpy as np
import torch
import os
import util
import model
from agent import Actor
import importlib

config = get_config()
os.environ['CUDA_VISIBLE_DEVICES'] = config['visible_device']
environment = importlib.import_module('environment.'+config['task_name'])

X_train, S1_train, S2_train, y_train, X_test, S1_test, S2_test, y_test = util.load_dataset('IL_dataset/'+config['task_name'])
ds = torch.load('Dynamics_Set/'+config['task_name'])
net = model.UVIN(ds).to('cuda')
Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
tracker = util.Tracker()

for epoch in range(config['max_training_epoch']):
    for i in range(0, X_train.shape[0], config['batch_size']):
        j = i + config['batch_size']
        if j > X_train.shape[0]: break
        optimizer.zero_grad()
        x = torch.cuda.FloatTensor(X_train[i:j]).permute(0,3,1,2)
        s1 = torch.cuda.LongTensor(S1_train[i:j])
        s2 = torch.cuda.LongTensor(S2_train[i:j])
        Q = net(x,s1,s2)
        label = torch.cuda.LongTensor(y_train[i:j].flatten())
        loss = Loss(Q.view(-1,config['A_size']),label)
        loss.backward()
        optimizer.step()
        tracker.writer.add_scalar("loss", float(loss.detach().cpu().numpy()), epoch*X_train.shape[0]+i)

model_path = os.path.join(tracker.path, 'model')
torch.save({'net':net.state_dict(),'ds':ds,'optimizer':optimizer.state_dict()}, model_path)

ac = 0
psum = 0
for i in range(0, X_test.shape[0], config['batch_size']):
    j = i + config['batch_size']
    if j > X_test.shape[0]: break
    x = torch.cuda.FloatTensor(X_test[i:j]).permute(0,3,1,2)
    s1 = torch.cuda.LongTensor(S1_test[i:j])
    s2 = torch.cuda.LongTensor(S2_test[i:j])
    Q = net(x,s1,s2)
    label = y_test[i:j].flatten()
    pre = Q.view(-1,config['A_size']).max(1)[1].cpu().numpy()
    ac += np.where(pre==label)[0].shape[0]
    psum += label.shape[0]
print('PA:', ac/psum)

agent = Actor(net, eps=False)
env = environment.Env()
sr,mean,std = util.test_game(env, agent, tracker, 0, config['test_final_game'])
print('test sr:%f, reward: mean:%f std:%f'%(sr, mean, std))