import gym
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from config import transitions
import actor_critic
import policy
from memory.tm.transitions_memory import TransitionsMemory as TM
from memory.rtm.reduced_transitions_memory import ReducedTransitionsMemory as RTM
from config.transitions import FrameTransition as ft 
from qrnn.q_lstm_gscale import QLSTMGSCALE
from environment.env import GymEnv
from action_exploration import OUActionNoise
from data import logger
from data import trial_logger as tl
from data import eval_logger as e_logger
from datetime import datetime
from collections import deque
import os
from policy import Epsilon
class COMPERDDPG(object):
    def __init__(self,task_name = "Pendulum-v1",log_base_dir="log") -> None:
        super().__init__()
        self.task_name = task_name
        self.env = GymEnv(self.task_name)
        self.tm = object
        self.rtm = object       
        self.qt = object
        self.critic_model = object
        self.actor_model = object
        self.target_actor = object
        self.target_critic = object
        self.critic_lr = 0.002#1e-3
        self.actor_lr = 0.001#1e-4
        self.gamma = 0.99       
        self.tau = 0.005#0.001
        self.noise_object = None
        self.log_base_dir = log_base_dir
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        self.train_log_path = "./"+self.log_base_dir+"/"+self.task_name+"/train/"
        self.eval_log_path = "./"+self.log_base_dir+"/"+self.task_name+"/eval/"
        self.checkpoint_path = "./"+self.log_base_dir+"/"+self.task_name+"/checkpoint/"
        self.epsilonInitial = 1.0
        self.epsilonFinal = 0.001     
        self.epsilonFraction = 0.20099

    def config_train_logger(self,trial):
        trial_path="trial"+str(trial)+"/"    
        logger.session(self.train_log_path+trial_path).__enter__()        

    def train_log(self,log_data_dict):
        for k, v in log_data_dict:
            logger.logkv(k, v)
        logger.dumpkvs()

    def config_eval_logger(self,trial):
        trial_path = "trial"+str(trial)+"/"    
        e_logger.session(self.eval_log_path).__enter__()

    def eval_log(self,log_data_dict):
        for k, v in log_data_dict:
            e_logger.logkv(k, v)
        e_logger.dumpkvs()

    def config_memories(self):               
        self.tm = TM(max_size=1000000,name="tm", memory_dir="./")
        self.rtm  = RTM(max_size=1000000,name="rtm",memory_dir="/.")
    
    def config_noise_object(self):
        std_dev = 0.2
        self.noise_object = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
    
    def __schedule_epsilon(self):
        self.epsilon = Epsilon(schedule_timesteps=int(self.epsilonFraction * 100000),initial_p=self.epsilonInitial,final_p=self.epsilonFinal)
   
    #@tf.function
    def update_actor_critic_nets2(self,state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor.model(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic.model([next_state_batch, target_actions], training=True)
            critic_value = self.critic_model.model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor_model.model(state_batch, training=True)
            critic_value = self.critic_model.model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.model.trainable_variables))

    def _get_transition_components(self,transitions):
        st_1 = transitions[:,:ft.T_IDX_ST_1[1]]
        a = transitions[:,ft.T_IDX_A[0]:ft.T_IDX_A[1]]
        r = transitions[:,ft.T_IDX_R] # get rewards
        st = transitions[:,ft.T_IDX_ST[0]:ft.T_IDX_ST[1]]
        q = transitions[:,ft.T_IDX_Q]# To ilustrate, but we do not need this here.
        done = transitions[:,ft.T_IDX_DONE] # get done signals

        #st_1 = transitions[:,:ft.T_IDX_ST_1[1]]
        #a = transitions[:,ft.T_IDX_A]
        #r = transitions[:,ft.T_IDX_R] # get rewards
        #st = transitions[:,ft.T_IDX_ST[0]:ft.T_IDX_ST[1]]
        #q = transitions[:,ft.T_IDX_Q]# To ilustrate, but we do not need this here.
        #done = transitions[:,ft.T_IDX_DONE] # get done signals
        return st_1,a,r,st,q,done
   
    def comput_loss_and_update2(self):
        transitions_batch=[]   
        if(self.rtm. __len__()>0):
            transitions_batch = self.rtm.sample_transitions_batch(64)
        else:
            transitions_batch = self.tm.sample_transitions_batch(64)
        
        st_1,a,r,st,q,done = self._get_transition_components(transitions_batch)
        #a = np.array(a)
        #a = a.reshape(a.shape[0],1)
        r = np.array(r)
        r = r.reshape(r.shape[0],1)
        state_batch = tf.convert_to_tensor(st_1)
        action_batch = tf.convert_to_tensor(a)
        reward_batch = tf.convert_to_tensor(r)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(st)

        self.update_actor_critic_nets2(state_batch, action_batch, reward_batch,next_state_batch)

        transitions = transitions_batch[:,:-2]            
        #target_predicted = qt.predict(transitions)
        #transitions = transitions.reshape(transitions.shape[0],1,transitions.shape[1])
        

        critic_value = self.critic_model.model([state_batch, action_batch], training=True).numpy()
        for i in range(len(st_1)):                 
            self.tm.add_transition(st_1[i],a[i],r[i],st[i],critic_value[i],float(done[i]))
        
        return state_batch, action_batch, reward_batch,next_state_batch,transitions

    #@tf.function
    def update_critic_target(self,state_batch, action_batch, reward_batch, next_state_batch,target_predicted,itr):
        with tf.GradientTape() as tape:           
            y = reward_batch + self.gamma * self.qt.predict(target_predicted)
            critic_value = self.critic_model.model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.model.trainable_variables))

    def update_critic_target2(self,state_batch, action_batch, reward_batch, next_state_batch,target_predicted,itr):
        with tf.GradientTape() as tape:
            actions = self.actor_model.model(state_batch, training=True)
            critic_value = self.critic_model.model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            y = reward_batch + self.gamma * self.qt.predict(target_predicted)
            #actor_loss = -tf.math.reduce_mean(critic_value)
            actor_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        actor_grad = tape.gradient(actor_loss, self.actor_model.model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.model.trainable_variables))
    
    def update_critic_target3(self,state_batch, action_batch, reward_batch, next_state_batch,target_predicted,itr):
        with tf.GradientTape() as tape:
            e = self.epsilon.value(itr)                      
            y =reward_batch + self.gamma * self.qt.predict(target_predicted)
            y = self.qt.predict(target_predicted)
            critic_value = self.critic_model.model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.model.trainable_variables))

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    #@tf.function
    def update_target(self,target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))


    def evaluate(self,trial,iterations,n_episodes=10):
        env = GymEnv(self.task_name)
        ep_reward_list = []
        for ep in range(n_episodes):
            ep_reward_list = []
            prev_state = env.reset()
            episodic_reward = 0
            done=False        
            while not done:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = policy.get_action_no_noise(tf_prev_state,self.actor_model.model,env.lower_bound,env.upper_bound)
                state, reward, done, info = env.step(action)
                episodic_reward += reward
                prev_state = state
                ep_reward_list.append(episodic_reward)
        
        avg_trial_rew = np.mean(ep_reward_list) if len(ep_reward_list)>0 else 0
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        print("Evaluate * {} * Eval Avg Reward is ==> {}".format(trial,avg_trial_rew))
        log_data_dict =[
            ('Trial',trial),
            ('Time',dt_string),
            ('Task',self.task_name),
            ('Int',iterations),
            ('EvalAvgRew',avg_trial_rew)]
        self.eval_log(log_data_dict)



    def train(self,tota_iterations=100,lstm_epochs=150,update_QTCritic_frequency=5,trainQTFreqquency=100,learningStartIter=1,q_lstm_bsize=1000,trial=1):
        
        self.__schedule_epsilon()       
        qlstm_log_path = "./"+self.log_base_dir+"/"+self.task_name+"/train/trial"+str(trial)    
        self.config_train_logger(trial)
        self.config_eval_logger(trial)
        self.actor_model = actor_critic.get_actor(self.task_name,self.env.num_states,self.env.upper_bound,self.env.num_actions)
        self.critic_model = actor_critic.get_critic(self.task_name,self.env.num_states,self.env.num_actions)

        self.target_actor = actor_critic.get_actor(self.task_name,self.env.num_states,self.env.upper_bound,self.env.num_actions)
        self.target_critic = actor_critic.get_critic(self.task_name,self.env.num_states,self.env.num_actions)

        # Making the weights equal initially
        self.target_actor.model.set_weights(self.actor_model.model.get_weights())
        self.target_critic.model.set_weights(self.critic_model.model.get_weights())

        self.config_memories()
        self.config_noise_object()

        #transitin_size = int((2*self.env.num_states + self.env.num_actions + 1))
        transitin_size=ft.T_LENGTH -2
        self.qt = QLSTMGSCALE(transitions_memory=self.tm,reduced_transitions_memory=self.rtm,inputshapex=1,inputshapey=transitin_size,outputdim=self.env.num_actions,
                                    verbose=False,transition_batch_size=q_lstm_bsize,netparamsdir='dev',target_optimizer="rmsprop",log_dir=qlstm_log_path,target_early_stopping=True)
        
        ep_reward_list = []        
        log_itr=0
        count=0
        first_qt_trained = False
        ep=0
        while (count<=tota_iterations):
            #prev_state = self.env.reset()
            prev_state = self.env.reset()[0]
            episodic_reward = 0
            itr = 1   
            run =True
            ep+=1
            while run:
                itr+=1
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = policy.get_action(tf_prev_state,self.actor_model.model,self.env.lower_bound,self.env.upper_bound,self.noise_object)
                state, reward, done,truncate, info = self.env.step(action)      
                
                q = self.critic_model.model([tf.convert_to_tensor(tf_prev_state), tf.convert_to_tensor(action)]).numpy()
                action = np.array(action)
                action = action.reshape(action.shape[1])
                #a = np.array(action)
                #a = a.reshape(a.shape[0],1)
                self.tm.add_transition(prev_state,action,reward,state,q,done)
                episodic_reward += reward       

                state_batch, action_batch, reward_batch,next_state_batch,transitions=self.comput_loss_and_update2()

                if (count % trainQTFreqquency == 0 and count > learningStartIter):
                    first_qt_trained=True
                    self.qt.train_q_prediction_withou_validation(n_epochs=lstm_epochs)

                if(first_qt_trained and (count % update_QTCritic_frequency == 0) and (count > learningStartIter)):                                    
                    self.update_critic_target(state_batch, action_batch, reward_batch,next_state_batch,transitions,count)
                
                if((count >1) and (count % 5000 == 0)):
                    self.evaluate(trial,count)
                    self.checkpoint_path = self.checkpoint_path+"trail"+str(trial)+"/"
                    self.actor_model.save_weights(self.checkpoint_path)
                
                self.update_target(self.target_actor.model.variables, self.actor_model.model.variables, self.tau)
                self.update_target(self.target_critic.model.variables, self.critic_model.model.variables, self.tau)
                    
                count+=1
                if done:            
                    run=False
                prev_state = state
                
                if(count % 200 == 0):
                    ep_reward_list.append(episodic_reward)                    
                    print("Episode * {} * Avg Reward is ==> {}".format(ep, np.mean(ep_reward_list[-50:])))
                    e =(1.0- self.epsilon.value(count))
                    avg_trial_rew = np.mean(ep_reward_list) if len(ep_reward_list)>0 else 0
                    avg_100_trial_rew = np.mean(ep_reward_list[-100:]) if len(ep_reward_list)>0 else 0
                    avg_40_trial_rew = np.mean(ep_reward_list[-50:]) if len(ep_reward_list)>0 else 0
                    avg_10_trial_rew = np.mean(ep_reward_list[-10:]) if len(ep_reward_list)>0 else 0
                    log_itr+=1
                    now = datetime.now()        
                    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
                    log_data_dict =[
                    ("Trial",trial),
                    ('LogCount',log_itr),
                    ('Task',self.task_name),
                    ('Time',dt_string),
                    ('TotalItr',count),
                    ('TMCount',self.tm.__len__()),
                    ('RTMCount',self.rtm.__len__()),
                    ('e',e),
                    ('Ep', ep),
                    ('EpItr', itr),
                    ("Done",done),
                    ('EpRew', episodic_reward),                    
                    ('AvgEp', avg_trial_rew),
                    ('Avg100Ep', avg_100_trial_rew),
                    ('AvgLast50Ep', avg_40_trial_rew),
                    ('AvgLast10Ep', avg_10_trial_rew)
                    ]
                    self.train_log(log_data_dict)


def config_trial_logger(base_log_dir = "./log/trials/"):    
        tl.session(base_log_dir).__enter__()

def trial_log(log_data_dict):
        for k, v in log_data_dict:
            tl.logkv(k, v)
        tl.dumpkvs()

def grid_search():
    task_name="Ant-v2"
    tota_iterations=[50000]
    lstm_epochs=[15]
    learningStartIter=[1]    
    trainQTFreqquency=[1]    
    update_QTCritic_frequency=[1]
    q_lstm_bsize=[50000]    
    trial=1
    max_trial =1
    log_base_dir ="log" 
    config_trial_logger(base_log_dir = "./"+log_base_dir+"/"+task_name+"/trials/")
    agent=None
    while trial<=max_trial:
        for tep in tota_iterations:
            for lstmep in lstm_epochs:
                for tqt in trainQTFreqquency:
                    for start in learningStartIter:
                        for upcritic in update_QTCritic_frequency:
                            for bs in q_lstm_bsize:                            
                                now = datetime.now()
                                dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
                                log_data_dict =[('Trial',trial),('Time',dt_string),('TotalEp',tep),
                                ('Tqt',tqt),('Lstmep',lstmep),('StartLearn',start),('Upcritic',upcritic),
                                ('Qlstm_bs',bs)]
                                trial_log(log_data_dict)
                                agent = COMPERDDPG(task_name=task_name,log_base_dir=log_base_dir)
                                agent.train(
                                    tota_iterations=tep,
                                    lstm_epochs=lstmep,
                                    trainQTFreqquency=tqt,
                                    learningStartIter= start,
                                    update_QTCritic_frequency=upcritic,
                                    q_lstm_bsize=bs,
                                    trial=trial)
        trial+=1                        

def main():
    grid_search()

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(type(e),e) 




            








