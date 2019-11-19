import torch
from config.config import get_config
import numpy as np

config = get_config()
imsize = config['imsize']
ckp = torch.load('IL_dataset/'+config['task_name'])
path = ckp['path']
goal = ckp['goal']

ds = np.zeros([config['S_size'],config['A_size'],config['C_size'],2], dtype=np.int)
pro_vector = np.array(range(config['P_size']), dtype=np.float32)
pro_vector = pro_vector / np.amax(pro_vector)

f = {}
for ma in range(len(path)):
    for step in range(len(path[ma])):
        sxy = path[ma][step][0]
        s = sxy[0]*imsize + sxy[1]
        a = path[ma][step][1]
        if step==len(path[ma])-1:
            s1xy = goal[ma]
        else:
            s1xy = path[ma][step+1][0]
        s1 = s1xy[0]*imsize + s1xy[1]
        if s not in f.keys():
            f[s] = {}
        if a not in f[s].keys():
            f[s][a] = {}
        if s1 not in f[s][a].keys():
            f[s][a][s1] = 0
        f[s][a][s1] += 1

for s in f.keys():
    for a in f[s].keys():
        tsum = sum(f[s][a].values())
        sclist = sorted(list(f[s][a].items()),key=lambda x:x[1],reverse=True)
        for i in range(min(ds.shape[2],len(sclist))):
            ds[s][a][i][0] = sclist[i][0]
            pid = np.argmin(np.absolute(pro_vector - sclist[i][1]/tsum))
            ds[s][a][i][1] = pid

torch.save(ds, 'Dynamics_Set/'+config['task_name'])