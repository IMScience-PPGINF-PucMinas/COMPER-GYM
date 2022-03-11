import gym
import numpy as np
from config.transitions import FrameTransition as ft
import numpy as np

class GymEnv(object):
    def __init__(self,task= "Pendulum-v1",verbose=True) -> None:
        super().__init__()
        self.verbose = verbose
        self.task =task
        self.gym_env = None
        self.num_states=None
        self.num_actions = None
        self.upper_bound=0
        self.lower_bound = 0
        self.__configenv__()
        self.__config_transitions__()
    
    def __configenv__(self):        
        self.gym_env = gym.make(self.task)
        self.num_states = self.gym_env.observation_space.shape[0]       
        self.num_actions = self.gym_env.action_space.shape[0]
        self.upper_bound = self.gym_env.action_space.high[0]
        self.lower_bound = self.gym_env.action_space.low[0]

        if(self.verbose):
            print("Environment Data for ->  {}".format(self.task))
            print("Size of State Space ->  {}".format(self.num_states))
            print("Size of Action Space ->  {}".format(self.num_actions))
            print("Size of Upper Bound of Action ->  {}".format(self.upper_bound))
            print("Size of Lower Bound of Action ->  {}".format(self.lower_bound))
    
    def __config_transitions__(self):              
        self.actions = self.gym_env.action_space
        self.nActions = self.actions.sample().shape[0]
        ft.ST_L= self.gym_env.observation_space.shape[0]
        ft.T_IDX_ST_1 = [0,ft.ST_L]
        ft.T_IDX_A    = [ft.ST_L,(ft.ST_L+self.nActions)]
        ft.T_IDX_R    = ft.T_IDX_A[1]
        ft.T_IDX_ST   = [ft.T_IDX_R+1,(ft.T_IDX_R+1+ft.ST_L)]
        ft.T_IDX_Q    = ft.T_IDX_ST[1]
        ft.T_IDX_DONE = ft.T_IDX_Q+1
        ft.T_LENGTH = ft.ST_L+self.nActions+1+ft.ST_L+1+1
        ft.T_N_IDX= ft.T_LENGTH-1
        
        #self.actions = self.gym_env.action_space
        #self.nActions = self.actions.sample().shape[0]
        #ft.ST_L= self.gym_env.observation_space.shape[0]
        #ft.T_IDX_ST_1 = [0,ft.ST_L]
        #ft.T_IDX_A    = ft.ST_L
        #ft.T_IDX_R    = ft.T_IDX_A+1
        #ft.T_IDX_ST   = [ft.T_IDX_R+1,(ft.T_IDX_R+1+ft.ST_L)]
        #ft.T_IDX_Q    = ft.T_IDX_ST[1]
        #ft.T_IDX_DONE = ft.T_IDX_Q+1
        #ft.T_LENGTH = ft.ST_L+1+1+ft.ST_L+1+1
        #ft.T_N_IDX= ft.T_LENGTH-1
        #temp=0
    
    def step(self,action):
        #action = np.array(action)
        #action = action[0]       
        return self.gym_env.step(action)

    def reset(self):
        return self.gym_env.reset()




