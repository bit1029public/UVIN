import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config.config import get_config
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class UVIN(nn.Module):

    def __init__(self, ds):
        super(UVIN, self).__init__()
        self.cf = get_config()
        self.ds = torch.cuda.LongTensor(ds)
        self.fre = {}
        self.fr = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=True)
        init_probability_vector = np.array(range(self.cf['P_size']), dtype=np.float32)
        init_probability_vector = init_probability_vector / np.amax(init_probability_vector)
        init_probability_vector = torch.cuda.FloatTensor(init_probability_vector)
        self.probability_vector = nn.Parameter(init_probability_vector, requires_grad=True)

        if self.cf['final_layer']=='linear':
            self.final_layer = nn.Linear(self.cf['A_size'], self.cf['A_size'])

    def forward(self, x, s1, s2):

        now_bs = x.shape[0]
        state_bs = s1.shape[1]
        sar = self.fr(x).view(now_bs, self.cf['imsize']**2, 1)
        sad = x[:,1,:,:].view(now_bs, self.cf['imsize']**2, 1, 1)/10.0
        
        p = torch.clamp(self.probability_vector, 0.0, 1.0)
        cf = self.ds[:,:,:,0].flatten()
        cp = torch.index_select(p,0,self.ds[:,:,:,1].flatten())
        cp = cp.view(1, self.cf['imsize']**2, self.cf['A_size'], self.cf['C_size'])
        v = torch.max(sar,dim=2)[0]

        for k in range(self.cf['VI_k']):
            vc = torch.index_select(v,1,cf)
            vc = vc.view(now_bs,self.cf['imsize']**2,self.cf['A_size'],self.cf['C_size'])
            q = sar + torch.sum(cp*self.cf['gamma']*vc*(1-sad),dim=3)
            v = torch.max(q,dim=2)[0]

        if self.cf['final_layer']=='linear':
            q = self.final_layer(q.view(-1,self.cf['A_size'])) + q.view(-1,self.cf['A_size'])
        q = q.view(now_bs, self.cf['imsize'], self.cf['imsize'], self.cf['A_size'])
        Q = torch.cuda.FloatTensor(np.zeros([now_bs,state_bs,self.cf['A_size']],dtype=np.float32))
        for i in range(now_bs):
            for j in range(state_bs):
                Q[i][j] = q[i][s1[i][j]][s2[i][j]]
        return Q

    def add_sample(self, s, a, s1):
        if s not in self.fre.keys():
            self.fre[s] = {}
        if a not in self.fre[s].keys():
            self.fre[s][a] = {}
        if s1 not in self.fre[s][a].keys():
            self.fre[s][a][s1] = 0
        self.fre[s][a][s1] += 1

    @ignore_warnings(category=ConvergenceWarning)
    def update_ds(self):
        idxs = []
        fre = []
        for s in self.fre.keys():
            for a in self.fre[s].keys():
                tsum = sum(self.fre[s][a].values())
                tlist = []
                for s1,cnt in self.fre[s][a].items():
                    tlist.append(([s,a,s1], cnt/tsum))
                tlist.sort(key=lambda x:x[1], reverse=True)
                for i in range(min(self.ds.shape[2],len(tlist))):
                    idxs.append(tlist[i][0]+[i])
                    fre.append(tlist[i][1])
        if len(fre)<self.cf['P_size']:
            return
        init = self.probability_vector.detach().cpu().numpy().reshape(-1,1)
        kmeans = KMeans(n_clusters=self.cf['P_size'],init=init,n_init=1)
        label = kmeans.fit_predict(np.array(fre).reshape([-1,1]))
        center = kmeans.cluster_centers_.reshape(-1)
        new_label = [(i,center[i]) for i in range(self.cf['P_size'])]
        new_label.sort(key=lambda x:x[1])
        new_label = [int(i[0]) for i in new_label]
        label = [new_label.index(i) for i in label]
        for i in range(len(idxs)):
            [s,a,s1,n] = idxs[i]
            s = s[0]*self.cf['imsize'] + s[1]
            s1 = s1[0]*self.cf['imsize'] + s1[1]
            self.ds[s][a][n][0] = s1
            self.ds[s][a][n][1] = int(label[i])