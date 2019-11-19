from config.config import get_config
import copy
import numpy as np
import torch
from random import shuffle, seed

def action2dxy(a,x,y):
    action = [0,1,2,3,4,5,6,7]
    se = x*y+2*x+y
    seed(int(se))
    shuffle(action)
    return {'0':[-1,0],'1':[1,0],'2':[0,1],'3':[0,-1],'4':[-1,1],'5':[-1,-1],'6':[1,1],'7':[1,-1]}.get(str(action[a]),[0,0])

class CoreEnv():

    def __init__(self):
        self.cf = get_config()
        self.reset()

    def step(self, action):
        dxy = action2dxy(action, self.current[0], self.current[1])
        self.current = self.current + dxy

    def reset(self):
        self.current = np.array([0,0], dtype=np.int)

    def set_state(self, state):
        self.current = np.array(state, dtype=np.int)

    def get_state(self):
        return tuple(self.current)

def build_asset():

    state_graph = {}
    vis = set()
    env = CoreEnv()
    imsize = get_config()['imsize']
    A_size = get_config()['A_size']

    def dfs(u, fa, faa):
        if not (0<=u[0] and u[0]<imsize and 0<=u[1] and u[1]<imsize):
            return
        if fa!=None:
            if fa not in state_graph.keys():
                state_graph[fa] = {}
            state_graph[fa][u] = faa
        if u in vis:
            return
        else:
            vis.add(u)
            print(u)
        for i in range(A_size):
            env.set_state(u)
            env.step(i)
            v = env.get_state()
            dfs(v, u, i)

    dfs((0,0),None,None)
    edge_count = 0
    for key in state_graph.keys():
        edge_count += len(state_graph[key].keys())
    torch.save({'state_graph':state_graph,'V_size':len(vis),'E_size':edge_count}, 'environment/asset/'+get_config()['task_name'])

class MapGenerator():

    def __init__(self, G):
        self.cf = get_config()
        self.G = G

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

        self.start = tuple(np.random.randint(low=0,high=self.cf['imsize'],size=2))
        self.now_map[self.start[0]][self.start[1]] = 0

        while True:
            self.goal = tuple(np.random.randint(low=0,high=self.cf['imsize'],size=2))
            if self.goal!=self.start:
                break
        self.now_map[self.goal[0]][self.goal[1]] = 0

        path = self.get_rand_path()
        for u in path:
            self.now_map[u[0]][u[1]] = 0

        return self.now_map, self.goal, self.start

def build_IL_dataset():

    config = get_config()
    G = torch.load('environment/asset/'+config['task_name'])['state_graph']
    mg = MapGenerator(G)

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
        self.mg = MapGenerator(G = torch.load('environment/asset/'+self.cf['task_name'])['state_graph'])
        self.core = CoreEnv()

    def reset(self):
        ma, goal, start = self.mg.get_map()
        self.map = np.zeros([self.cf['imsize'], self.cf['imsize'], 2], dtype=np.uint8)
        self.goal = goal
        self.map[:,:,0] = ma
        self.map[goal[0],goal[1],1] = 10
        self.left_step = 250
        self.core.set_state(start)
        return [self.map, self.core.get_state()]
        
    def step(self, action):

        self.core.step(action)
        xy = self.core.get_state()

        if not (0<=xy[0] and xy[0]<self.cf['imsize'] and 0<=xy[1] and xy[1]<self.cf['imsize']):
            return [self.map, (0,0)], -1.0, True, None

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