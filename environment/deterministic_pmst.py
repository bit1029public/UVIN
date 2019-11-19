import random
import numpy as np
import torch
import copy
from config.config import get_config
from random import shuffle

h2op_list = ['get8_log', 'get13_cobblestone', 'get29_planks', 'get8_stick', 
             'get5_crafting_table', 'get1_wooden_axe', 'get1_wooden_pickaxe', 
             'get1_furnace', 'get1_stone_axe', 'get1_stone_pickaxe']
random.seed(61)
random.shuffle(h2op_list)
# ['get13_cobblestone', 'get8_log', 'get1_stone_pickaxe', 
# 'get5_crafting_table', 'get1_wooden_axe', 'get1_stone_axe', 
# 'get1_wooden_pickaxe', 'get8_stick', 'get29_planks', 
# 'get1_furnace']

h2state = ['log', 'cobblestone', 'planks', 'stick', 
           'crafting_table', 'wooden_axe', 'wooden_pickaxe', 
           'furnace', 'stone_axe', 'stone_pickaxe']
random.seed(23)
random.shuffle(h2state)
# ['furnace', 'stone_axe', 'wooden_axe', 'wooden_pickaxe', 'stick', 
# 'planks', 'stone_pickaxe', 'log', 'cobblestone', 'crafting_table']

class CoreEnv():

    def __init__(self):
        self.reset()

    def reset(self):
        self.now_state = np.zeros([len(h2state)], dtype=np.bool)

    def get_state(self):
        return tuple(self.now_state)

    def set_state(self, state):
        self.now_state = np.array(state, dtype=np.bool)

    def step(self, h2op):

        op = h2op_list[h2op]
        if op=='get8_log':
            if not self.now_state[h2state.index('cobblestone')]:
                self.now_state[h2state.index('log')] = True
        elif op=='get13_cobblestone':
            self.now_state[h2state.index('cobblestone')] = True
        elif op=='get29_planks':
            if self.now_state[h2state.index('log')]:
                self.now_state[h2state.index('planks')] = True
        elif op=='get8_stick':
            if self.now_state[h2state.index('planks')]:
                self.now_state[h2state.index('stick')] = True
        elif op=='get5_crafting_table':
            if self.now_state[h2state.index('planks')]:
                self.now_state[h2state.index('crafting_table')] = True
        elif op=='get1_wooden_axe':
            if self.now_state[h2state.index('crafting_table')]\
               and self.now_state[h2state.index('planks')]\
               and self.now_state[h2state.index('stick')]:
                self.now_state[h2state.index('wooden_axe')] = True
        elif op=='get1_wooden_pickaxe':
            if self.now_state[h2state.index('crafting_table')]\
               and self.now_state[h2state.index('planks')]\
               and self.now_state[h2state.index('stick')]:
                self.now_state[h2state.index('wooden_pickaxe')] = True
        elif op=='get1_furnace':
            if self.now_state[h2state.index('crafting_table')]\
               and self.now_state[h2state.index('cobblestone')]:
                self.now_state[h2state.index('furnace')] = True
        elif op=='get1_stone_axe':
            if self.now_state[h2state.index('crafting_table')]\
               and self.now_state[h2state.index('cobblestone')]\
               and self.now_state[h2state.index('stick')]:
                self.now_state[h2state.index('stone_axe')] = True
        elif op=='get1_stone_pickaxe':
            if self.now_state[h2state.index('crafting_table')]\
               and self.now_state[h2state.index('cobblestone')]\
               and self.now_state[h2state.index('stick')]:
                self.now_state[h2state.index('stone_pickaxe')] = True

def s2xy(state):
    num = 0
    for i in range(len(state)):
        num += int(state[i]) * (2**i)
    return (num//32,num%32)

def build_asset():

    state_graph = {}
    vis = set()
    env = CoreEnv()

    def dfs(u, fa, faop):
        if u==fa:
            return
        vis.add(u)
        print(u)
        if fa!=None:
            if fa not in state_graph.keys():
                state_graph[fa] = {}
            state_graph[fa][u] = faop
        
        for i in range(len(h2op_list)):
            env.set_state(u)
            env.step(i)
            v = env.get_state()
            dfs(v, u, i)

    dfs(tuple(env.now_state), None, None)
    edge_count = 0
    for key in state_graph.keys():
        edge_count += len(state_graph[key].keys())
    V = [s2xy(s) for s in vis]
    G = {}
    for s in state_graph.keys():
        G[s2xy(s)] = {}
        for s1 in state_graph[s].keys():
            G[s2xy(s)][s2xy(s1)] = state_graph[s][s1]
    torch.save({'state_graph':G,'V':V,'V_size':len(V),'E_size':edge_count}, 'environment/asset/'+get_config()['task_name'])

class MapGenerator():

    def __init__(self, G, V):
        self.cf = get_config()
        self.G = G
        self.V = V

    def rand_dfs(self, u):
        if u==self.goal:
            return [u]
        if u in self.tmpvis:
            return []
        self.tmpvis.add(u)
        if u not in self.G.keys():
            return []
        vlist = list(self.G[u].keys())
        shuffle(vlist)
        for v in vlist:
            res = self.rand_dfs(v)
            if len(res)!=0:
                return [u]+res
        return []

    def get_rand_path(self):
        self.tmpvis = set()
        return self.rand_dfs(self.start)[:-1]

    def get_shortest_path(self):
        q = [self.start]
        fa = {}
        faop = {}
        tmpvis = set()
        tmpvis.add(self.start)
        while len(q)!=0:
            u = q.pop(0)
            if u==self.goal:
                break
            if u not in self.G.keys():
                continue
            valist = sorted(list(self.G[u].items()),key=lambda x:x[1])
            for (v,a) in valist:
                if v not in tmpvis:
                    tmpvis.add(v)
                    if self.now_map[v[0]][v[1]]==0:
                        q.append(v)
                        fa[v] = u
                        faop[v] = a

        res = []
        while(True):
            if u==self.start:
                break
            res.append((fa[u], faop[u]))
            u = fa[u]
        res.reverse()
        return res

    def get_map(self):
        self.now_map = np.zeros([self.cf['imsize'], self.cf['imsize']], dtype=np.uint8)

        ob_num = np.random.randint(low=0,high=(self.cf['imsize']**2)//2)
        for _ in range(ob_num):
            ob = tuple(np.random.randint(low=0,high=self.cf['imsize'],size=2))
            self.now_map[ob[0]][ob[1]] = 1

        self.start = (0,0)
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
    mg = MapGenerator(ckp['state_graph'], ckp['V'])

    num_map = 20000
    data_pre_map = 10
    imsize = config['imsize']

    X = np.zeros([num_map, imsize, imsize, 2], dtype=np.uint8)
    S1 = np.zeros([num_map, data_pre_map], dtype=np.uint8)
    S2 = np.zeros([num_map, data_pre_map], dtype=np.uint8)
    y = np.zeros([num_map, data_pre_map], dtype=np.uint8)
    path_data = []
    goal_data = []

    for ma in range(num_map):
        print('start generate %d'%ma)
        now_map, goal, _ = mg.get_map()
        X[ma,:,:,0] = now_map
        X[ma,goal[0],goal[1],1] = 10
        path = mg.get_shortest_path()
        path_data.append(path)
        goal_data.append(goal)
        for i in range(data_pre_map):
            idx = np.random.randint(low=0,high=len(path))
            S1[ma,i] = path[idx][0][0]
            S2[ma,i] = path[idx][0][1]
            y[ma,i] = path[idx][1]

    torch.save({'X':X,'S1':S1,'S2':S2,'y':y,'path':path_data,'goal':goal_data}, 'IL_dataset/'+config['task_name'])

class Env():

    def __init__(self):
        self.cf = get_config()
        ckp = torch.load('environment/asset/'+self.cf['task_name'])
        self.mg = MapGenerator(ckp['state_graph'], ckp['V'])
        self.core = CoreEnv()

    def reset(self):
        ma, goal, start = self.mg.get_map()
        self.map = np.zeros([self.cf['imsize'], self.cf['imsize'], 2], dtype=np.uint8)
        self.goal = goal
        self.map[:,:,0] = ma
        self.map[goal[0],goal[1],1] = 10
        self.left_step = 250
        self.core.reset()
        return [self.map, s2xy(self.core.get_state())]
        
    def step(self, action):

        self.core.step(action)
        xy = s2xy(self.core.get_state())

        if xy==self.goal:
            r, d = 1.0, True
        elif self.map[xy[0],xy[1],0]==1:
            r, d = -1.0, True
        else:
            r, d = 0.0, False

        self.left_step -= 1
        if self.left_step<=0:
            d = True
        
        return [self.map, xy], r, d, None