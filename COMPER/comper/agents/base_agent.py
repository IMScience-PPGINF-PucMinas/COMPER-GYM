from datetime import datetime
from comper.data import logger
import gym
from comper.dnn.qmlp import MlpNet
from comper.config import exceptions as ex
from comper.config import parameters as param
from numpy.random import RandomState
from comper.config.transitions import FrameTransition as ft
from comper.config.transitions import FrameTransitionTypes as f_type

class BaseAgent(object):
    def __init__(self,rom_name,log_dir,nets_param_dir,display_screen,run_type=param.RunType.NO_SET):
        self.run_type =run_type
        self.rom_name = rom_name
        self.env = {}
        self.statePreprocessor={}
        self.logDir = ""
        self.nets_param_dir= ""
        self.nActions=0                     
        self.displayScreen = display_screen
        self.__validate_base_parameters(display_screen)
        self.__validate_run_type()        
        self.__set_nets_param_dir(nets_param_dir)
        self.__set_logs_dir(log_dir)        
        self.__init_logger()
        self.config_environment()       

    def __validate_run_type(self):
        if(self.run_type==param.RunType.NO_SET):
            raise ex.ExceptionRunType(self.run_type)

    def __validate_base_parameters(self,display_screen):        
        if display_screen != param.DisplayScreen.ON and display_screen !=param.DisplayScreen.OFF:
            raise ex.ExceptionDisplayScreen(display_screen)   
    
    def __set_nets_param_dir(self,nets_param_dir=""):
        self.nets_param_dir =  "./netparams_comper/"+nets_param_dir
        

    def __set_logs_dir(self,log_dir):
        base_dir = "./log_comper/" 
        if self.run_type==param.RunType.TRAIN:
            base_dir = base_dir+"train/"        
        elif self.run_type==param.RunType.TEST:
            base_dir = base_dir+"test/"        
        self.logDir = base_dir+"/"+log_dir
             
        

    def __init_logger(self):
        logger.session(self.logDir).__enter__()
    
    def log(self,log_data_dict):
        for k, v in log_data_dict:
            logger.logkv(k, v)
        logger.dumpkvs()

    def reset_env(self):
        return self.env.reset()

    def config_environment(self,):        
        self.env = gym.make(self.rom_name)        
        self.actions = self.env.action_space
        self.nActions = self.actions.sample().shape[0]
        ft.ST_L= self.env.observation_space.shape[0]
        ft.T_IDX_ST_1 = [0,ft.ST_L]
        ft.T_IDX_A    = ft.ST_L
        ft.T_IDX_R    = ft.T_IDX_A+1
        ft.T_IDX_ST   = [ft.T_IDX_R+1,(ft.T_IDX_R+1+ft.ST_L)]
        ft.T_IDX_Q    = ft.T_IDX_ST[1]
        ft.T_IDX_DONE = ft.T_IDX_Q+1
        ft.T_LENGTH = ft.ST_L+1+1+ft.ST_L+1+1
        ft.T_N_IDX= ft.T_LENGTH-1
        
class BaseTrainAgent(BaseAgent):
    def __init__(self,rom_name,maxtotalframes,frames_ep_decay,train_frequency,update_target_frequency,learning_start_iter,log_frequency,
                 log_dir,nets_param_dir,memory_dir,save_states_frq,persist_memories,save_networks_weigths=True,
                 save_networks_states=0,display_screen=False):
        
        super().__init__(rom_name,log_dir,nets_param_dir,display_screen,run_type=param.RunType.TRAIN)
              
        
        self.maxtotalframes = maxtotalframes
        self.frames_ep_decay = frames_ep_decay
        self.trainQFrequency = train_frequency
        self.trainQTFreqquency = update_target_frequency
        self.learningStartIter = learning_start_iter
        self.logFrequency = log_frequency
        self.save_states_freq = save_states_frq
        self.persist_memories = persist_memories
        self.save_networks_weigths = save_networks_weigths
        self.save_networks_states = save_networks_states
        self.randon = RandomState(123)
        self.target_optimizer = "rmsprop"
        self.memory_dir=""

        self.__set_memory_dir(memory_dir)
        self.__validate_train_parameters()

    def __set_memory_dir(self,memory_dir):
        self.memory_dir="./transitions_memory/"+memory_dir
        
    def __validate_train_parameters(self):
        if self.persist_memories != param.PersistMemory.YES and self.persist_memories != param.PersistMemory.NO:
            raise ex.ExceptionPersistMemory(self.persist_memories)
           
class BaseTestAgent(BaseAgent):
    def __init__(self, rom_name,rom_file_path,log_dir,nets_param_dir,display_screen=param.DisplayScreen.OFF,framesmode=param.FrameMode.SINGLE):
        super().__init__(rom_name, rom_file_path, log_dir, nets_param_dir, framesmode, display_screen,param.RunType.TEST)        
        
        self.q_input_shape = (ft.ST_W, ft.ST_H, ft.ST_L)
        self.q = {}
        self.trialReward =0
        self.sumTrialsRewards=0
        self.n_trials=1
        self.trial_count=0
        self.total_frames=0

    def initialize(self):
        print("Init test agent with "+self.framesmode+"frames")
        self.__config_environment()
        self.__create_q_network()

    def __config_environment(self):        
        super().config_environment(repeat_action_probability=0,frame_skip=5)
    
    def __create_q_network(self):
        self.q = MlpNet(netparamsdir=self.nets_param_dir,run_type=param.RunType.TEST)
        self.q.create(input_shape = self.q_input_shape,output_dim=self.nActions)
        self.q.compile_model()

    def LogTestData(self,current_trial,trial_reward,n_frames,sum_trials_rewards,total_frames):
        now = datetime.now()        
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        log_data_dict =[('Rom',self.rom_name),
        ('Time',dt_string),
        ('Trial', current_trial),
        ('TrialReward', trial_reward),
        ('TrialFrames', n_frames),
        ('SumTrialsRewads', sum_trials_rewards),
        ('TotalFrames', total_frames)]
        super().log(log_data_dict) 
    
   
    


       




