from config.config import get_config
import numpy as np
import random
import torch
import copy

inventory_list = ['log', 'planks', 'stick', 'crafting_table', 'wooden_axe', 'wooden_pickaxe']
inven_max = np.array([4, 15, 4, 2, 1, 1], dtype=np.int)

log_num_list = [0, 1, 2, 3]
log_pro = [0.1, 0.6, 0.2, 0.1]
place_sr = 0.8

option_list = [
                'obtain_log',# 0
                'obtain_planks',#1
                'obtain_stick',#2
                'obtain_crafting_table',#3
                'obtain_wooden_axe',#4
                'obtain_wooden_pickaxe'#5
            ]

class CoreEnv():

    def __init__(self):
        self.reset()

    def reset(self):
        self.now_inven = np.zeros([len(inventory_list)], dtype=np.int)

    def get_state(self):
        return tuple(self.now_inven)

    def set_state(self, inven):
        self.now_inven = np.array(inven, dtype=np.int)

    # log_num only use in build graph
    def step(self, idx, log_num=-1):

        op = option_list[idx]

        if op =='obtain_log':
            if log_num==-1:
                self.now_inven[inventory_list.index('log')] += np.random.choice(a=len(log_pro), p=log_pro)
            else:
                self.now_inven[inventory_list.index('log')] += log_num
        elif op == 'obtain_planks':
            if self.now_inven[inventory_list.index('log')] > 0:
                self.now_inven[inventory_list.index('log')] -= 1
                self.now_inven[inventory_list.index('planks')] += 4
        elif op == 'obtain_stick':
            if self.now_inven[inventory_list.index('planks')] >= 2:
                self.now_inven[inventory_list.index('planks')] -= 2
                self.now_inven[inventory_list.index('stick')] += 4
        elif op == 'obtain_crafting_table':
            if self.now_inven[inventory_list.index('planks')] >= 4:
                self.now_inven[inventory_list.index('planks')] -= 4
                self.now_inven[inventory_list.index('crafting_table')] += 1
        elif op == 'obtain_wooden_axe':
            if self.now_inven[inventory_list.index('stick')] >= 2 \
               and self.now_inven[inventory_list.index('planks')] >= 2 \
               and self.now_inven[inventory_list.index('crafting_table')] >= 1 \
               and random.random() < place_sr:
                self.now_inven[inventory_list.index('stick')] -= 2
                self.now_inven[inventory_list.index('planks')] -= 2
                self.now_inven[inventory_list.index('crafting_table')] -= 1
                self.now_inven[inventory_list.index('wooden_axe')] += 1
        elif op == 'obtain_wooden_pickaxe':
            if self.now_inven[inventory_list.index('stick')] >= 2 \
               and self.now_inven[inventory_list.index('planks')] >= 3 \
               and self.now_inven[inventory_list.index('crafting_table')] >= 1 \
               and random.random() < place_sr:
                self.now_inven[inventory_list.index('stick')] -= 2
                self.now_inven[inventory_list.index('planks')] -= 3
                self.now_inven[inventory_list.index('crafting_table')] -= 1
                self.now_inven[inventory_list.index('wooden_pickaxe')] += 1

class S2xy_mapper():

    def __init__(self, inven_set):
        self.inven_set = inven_set
        self.imsize = get_config()['imsize']

    def __call__(self, s):
        num = self.inven_set.index(s)
        return (num//self.imsize,num%self.imsize)

def build_asset():

    global place_sr
    place_sr = 1.0
    state_graph = {}
    vis = set()
    env = CoreEnv()

    def dfs(u, fa, faop):

        if fa!=None and fa!=u:
            if fa not in state_graph.keys():
                state_graph[fa] = {}
            state_graph[fa][u] = faop

        if u in vis:
            return
        else:
            vis.add(u)
            print(u)

        if np.where(np.array(u,dtype=np.int)>inven_max)[0].shape[0] > 0:
            return

        # option 0 : get log
        for ln in log_num_list:
            env.set_state(u)
            env.step(0, log_num=ln)
            v = env.get_state()
            dfs(v, u, 0)
        
        for i in range(1, len(option_list)):
            env.set_state(u)
            env.step(i)
            v = env.get_state()
            dfs(v, u, i)

    root = tuple([0]*len(inventory_list))
    dfs(root, None, None)

    s2xy = S2xy_mapper(list(vis))
    V = []
    border_V = []
    for s in vis:
        if np.where(np.array(s,dtype=np.int)>inven_max)[0].shape[0]>0:
            border_V.append(s2xy(s))
        else:
            V.append(s2xy(s))
    G = {}
    for s in state_graph.keys():
        G[s2xy(s)] = {}
        for s1 in state_graph[s].keys():
            G[s2xy(s)][s2xy(s1)] = state_graph[s][s1]

    imsize = get_config()['imsize']
    S_size = get_config()['imsize']**2
    A_size = get_config()['A_size']
    T = np.zeros([S_size, A_size, S_size], dtype=np.float32)
    for s in G.keys():
        for s1 in G[s].keys():
            sn = s[0]*imsize + s[1]
            s1n = s1[0]*imsize + s1[1]
            if G[s][s1]==0:
                delta = np.array(s2xy.inven_set[s1n],dtype=np.int) - np.array(s2xy.inven_set[sn],dtype=np.int)
                dlog = delta[0]
                T[sn][0][s1n] = log_pro[dlog]
                T[sn][0][sn] = log_pro[0]
            elif G[s][s1]==4 or G[s][s1]==5:
                T[sn][G[s][s1]][s1n] = 0.8
                T[sn][G[s][s1]][sn] = 0.2
            else:
                T[sn][G[s][s1]][s1n] = 1.0
    for s in range(T.shape[0]):
        for a in range(T.shape[1]):
            sump = np.sum(T[s,a,:])
            assert(np.isclose(sump,0) or np.isclose(sump,1))
            if sump==0:
                T[s,a,s] = 1.0

    torch.save({'state_graph':G,'V':V,'inven_set':s2xy.inven_set,'T':T,'border_V':border_V}, 'environment/asset/'+get_config()['task_name'])

def value_iteration(R,T,done):

    V = np.zeros([T.shape[0]],dtype=np.float32)
    V1 = np.zeros([T.shape[0]],dtype=np.float32)
    gamma = get_config()['gamma']

    while True:
        for s in range(T.shape[0]):
            if done[s]!=0:
                V1[s] = R[s]
                continue
            else:
                V1[s] = np.max(np.sum(gamma*V*T[s,:,:], axis=1))
        if np.isclose(np.amax(np.abs(V-V1)),0.0):
            break
        V = copy.deepcopy(V1)

    policy = np.zeros([T.shape[0]],dtype=np.uint8)
    for s in range(T.shape[0]):
        policy[s] = np.argmax(np.sum(T[s,:,:]*V1, axis=1))
    return policy, V1

class MapGenerator():

    def __init__(self, G, V, T, s2xy, border_V):
        self.cf = get_config()
        self.G = G
        self.V = V
        self.s2xy = s2xy
        self.T = T
        self.border_V = border_V

    def rand_dfs(self, u):
        if u==self.goal:
            return [u]
        if u in self.tmpvis:
            return []
        self.tmpvis.add(u)
        if u not in self.G.keys():
            return []
        vlist = list(self.G[u].keys())
        random.shuffle(vlist)
        for v in vlist:
            if v in self.V:
                res = self.rand_dfs(v)
                if len(res)!=0:
                    return [u]+res
        return []

    def get_rand_path(self):
        self.tmpvis = set()
        return self.rand_dfs(self.start)[:-1]

    def get_shortest_path(self):
        imsize = self.cf['imsize']
        R = self.now_map.reshape(-1).astype(np.float32) * -1.0
        R[self.goal[0]*imsize+self.goal[1]] = 1.0
        done = np.zeros([self.T.shape[0]], dtype=np.float32)
        for s in range(self.T.shape[0]):
            if R[s]!=0:
                done[s] = 1.0
        for (x,y) in self.border_V:
            done[x*imsize+y] = 1.0
        policy, Vfunction = value_iteration(R, self.T, done)

        res = []
        now = self.start
        nown = now[0]*self.cf['imsize'] + now[1]
        while True:
            res.append((now,policy[nown]))
            nown = np.random.choice(a=self.T.shape[0],p=self.T[nown,policy[nown],:])
            now = (nown//imsize,nown%imsize)
            if done[nown]!=0:
                break
            nz = np.where(self.T[nown,policy[nown],:]!=0)[0]
            if nz[0]==nown and nz.shape[0]==1:
                break
        return res, now

    def get_map(self):
        self.now_map = np.zeros([self.cf['imsize'], self.cf['imsize']], dtype=np.uint8)

        ob_num = np.random.randint(low=0,high=(self.cf['imsize']**2)//4)
        for _ in range(ob_num):
            ob = tuple(np.random.randint(low=0,high=self.cf['imsize'],size=2))
            self.now_map[ob[0]][ob[1]] = 1

        self.start = self.s2xy(tuple([0]*self.cf['A_size']))
        self.now_map[self.start[0]][self.start[1]] = 0

        while True:
            self.goal = self.V[np.random.randint(low=0,high=len(self.V))]
            if self.goal!=self.start:
                break
        self.now_map[self.goal[0]][self.goal[1]] = 0

        path = self.get_rand_path()
        for u in path:
            self.now_map[u[0]][u[1]] = 0

        return self.now_map, self.goal, self.start

def build_IL_dataset():

    config = get_config()
    ckp = torch.load('environment/asset/'+config['task_name'])
    mg = MapGenerator(ckp['state_graph'], ckp['V'], ckp['T'], S2xy_mapper(ckp['inven_set']), ckp['border_V'])

    num_map = 20000
    data_pre_map = 10
    imsize = config['imsize']

    X = np.zeros([num_map, imsize, imsize, 2], dtype=np.uint8)
    S1 = np.zeros([num_map, data_pre_map], dtype=np.uint8)
    S2 = np.zeros([num_map, data_pre_map], dtype=np.uint8)
    y = np.zeros([num_map, data_pre_map], dtype=np.uint8)
    path_data = []
    end_data = []

    succ = 0
    for ma in range(num_map):
        print('start generate %d'%ma)
        now_map, goal, _ = mg.get_map()
        X[ma,:,:,0] = now_map
        X[ma,goal[0],goal[1],1] = 10
        path, end = mg.get_shortest_path()
        path_data.append(path)
        end_data.append(end)
        for i in range(data_pre_map):
            idx = np.random.randint(low=0,high=len(path))
            S1[ma,i] = path[idx][0][0]
            S2[ma,i] = path[idx][0][1]
            y[ma,i] = path[idx][1]

        if end==goal:
            succ += 1

    torch.save({'X':X,'S1':S1,'S2':S2,'y':y,'path':path_data,'goal':end_data}, 'IL_dataset/'+config['task_name'])
    print(succ)

class Env():

    def __init__(self):
        self.cf = get_config()
        ckp = torch.load('environment/asset/'+self.cf['task_name'])
        self.s2xy = S2xy_mapper(ckp['inven_set'])
        self.mg = MapGenerator(ckp['state_graph'], ckp['V'], ckp['T'], self.s2xy, ckp['border_V'])
        self.core = CoreEnv()

    def reset(self):
        ma, goal, start = self.mg.get_map()
        self.map = np.zeros([self.cf['imsize'], self.cf['imsize'], 2], dtype=np.uint8)
        self.goal = goal
        self.map[:,:,0] = ma
        self.map[goal[0],goal[1],1] = 10
        self.left_step = 250
        self.core.reset()
        return [self.map, self.s2xy(self.core.get_state())]
        
    def step(self, action):

        self.core.step(action)
        xy = self.s2xy(self.core.get_state())

        if xy==self.goal:
            r, d = 1.0, True
        elif self.map[xy[0],xy[1],0]==1:
            r, d = -1.0, True
        else:
            r, d = 0.0, False

        self.left_step -= 1
        if self.left_step<=0 or np.where(self.core.now_inven>inven_max)[0].shape[0]>0:
            d = True
        
        return [self.map, xy], r, d, None