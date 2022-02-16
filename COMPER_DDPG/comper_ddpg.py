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
from data import logger
from data import trial_logger as tl
from data import eval_logger as e_logger
from datetime import datetime
import os
class COMPERDDPG(object):
    def __init__(self,task_name = "Pendulum-v1") -> None:
        super().__init__()
        self.task_name = task_name
        self.env = GymEnv(self.task_name)
        self.tm = object
        self.rtm = object
        self.critic_model = object
        self.qt = object
        self.actor_model = object
        self.target_actor = object
        self.target_critic = object
        self.critic_lr = 0.002
        self.actor_lr = 0.001
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)             
        self.gamma = 0.99       
        self.tau = 0.005


    def config_logger(self,base_log_dir = "./log/train/"):    
        logger.session(base_log_dir).__enter__()
        

    def log(self,log_data_dict):
        for k, v in log_data_dict:
            logger.logkv(k, v)
        logger.dumpkvs()

    def config_eval_logger(self,base_log_dir = "./log/eval/"):    
        e_logger.session(base_log_dir).__enter__()

    def eval_log(self,log_data_dict):
        for k, v in log_data_dict:
            e_logger.logkv(k, v)
        e_logger.dumpkvs()

    def config_memories(self):               
        self.tm = TM(max_size=100000,name="tm", memory_dir="./")
        self.rtm  = RTM(max_size=100000,name="rtm",memory_dir="/.")
            

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    #@tf.function
    def update_actor_critic_nets(self,state_batch, action_batch, reward_batch, next_state_batch,target_predicted):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.    
        with tf.GradientTape() as tape:
            #target_actions = target_actor.model(next_state_batch, training=True)
            #y = reward_batch + gamma * target_critic.model([next_state_batch, target_actions], training=True)        
            y = reward_batch + self.gamma * self.qt.lstm.lstm(target_predicted)
            #y = reward_batch + gamma * target_predicted
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

    # We compute the loss and update parameters
    def comput_loss_and_update(self):
        transitions_batch=[]   
        if(self.rtm. __len__()>0):
            transitions_batch = self.rtm.sample_transitions_batch(64)
        else:
            transitions_batch = self.tm.sample_transitions_batch(64)
        
        st_1,a,r,st,q,done = self._get_transition_components(transitions_batch)
        a = np.array(a)
        a = a.reshape(a.shape[0],1)
        r = np.array(r)
        r = r.reshape(r.shape[0],1)
        state_batch = tf.convert_to_tensor(st_1)
        action_batch = tf.convert_to_tensor(a)
        reward_batch = tf.convert_to_tensor(r)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(st)
        transitions = transitions_batch[:,:-2]            
        #target_predicted = qt.predict(transitions)
        transitions = transitions.reshape(transitions.shape[0],1,transitions.shape[1])
        self.update_actor_critic_nets(state_batch, action_batch, reward_batch,next_state_batch,transitions)

        critic_value = self.critic_model.model([state_batch, action_batch], training=True).numpy()
        for i in range(len(st_1)):                 
            self.tm.add_transition(st_1[i],a[i],r[i],st[i],critic_value[i],float(done[i]))

    def comput_loss_and_update2(self):
        transitions_batch=[]   
        if(self.rtm. __len__()>0):
            transitions_batch = self.rtm.sample_transitions_batch(64)
        else:
            transitions_batch = self.tm.sample_transitions_batch(64)
        
        st_1,a,r,st,q,done = self._get_transition_components(transitions_batch)
        a = np.array(a)
        a = a.reshape(a.shape[0],1)
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
    def update_critic_target(self,state_batch, action_batch, reward_batch, next_state_batch,target_predicted):
        with tf.GradientTape() as tape:
            #target_actions = target_actor.model(next_state_batch, training=True)
            #y = reward_batch + gamma * target_critic.model([next_state_batch, target_actions], training=True)        
            #y = reward_batch + self.gamma * self.qt.lstm.lstm(target_predicted)
            #y = reward_batch + gamma * target_predicted
            y =reward_batch + self.gamma * self.qt.predict(target_predicted)

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



    def train(self,total_episodes=100,lstm_epochs=150,update_QTCritic_frequency=5,trainQTFreqquency=100,learningStartIter=1,q_lstm_bsize=1000,trial=1):
        logFrequency=100
        base_log_dir="./log/train/"
        rew_log_path = base_log_dir+"trial"+str(trial)+"/"
        qlstm_log_path = rew_log_path+"lstm/"    
        self.config_logger(rew_log_path)
        self.config_eval_logger()
        self.actor_model = actor_critic.get_actor(self.env.num_states,self.env.upper_bound)
        self.critic_model = actor_critic.get_critic(self.env.num_states,self.env.num_actions)

        self.target_actor = actor_critic.get_actor(self.env.num_states,self.env.upper_bound)
        self.target_critic = actor_critic.get_critic(self.env.num_states,self.env.num_actions)

        # Making the weights equal initially
        self.target_actor.model.set_weights(self.actor_model.model.get_weights())
        self.target_critic.model.set_weights(self.critic_model.model.get_weights())

        self.config_memories()
        #transitin_size = int((2*self.env.num_states + self.env.num_actions + 1))
        transitin_size=ft.T_LENGTH -2
        self.qt = QLSTMGSCALE(transitions_memory=self.tm,reduced_transitions_memory=self.rtm,inputshapex=1,inputshapey=transitin_size,outputdim=self.env.num_actions,
                                    verbose=False,transition_batch_size=q_lstm_bsize,netparamsdir='dev',target_optimizer="rmsprop",log_dir=qlstm_log_path,target_early_stopping=True)
        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        log_itr=0
        count=0
        first_qt_trained = False
        for ep in range(total_episodes):

            prev_state = self.env.reset()
            episodic_reward = 0
            itr = 1   
            run =True
            while run:
                itr+=1
                # Uncomment this to see the Actor in action
                # But not in a python notebook.
                # env.render()

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = policy.get_action(tf_prev_state,self.actor_model.model,self.env.lower_bound,self.env.upper_bound)
                # Recieve state and reward from environment.
                state, reward, done, info = self.env.step(action)      
                a = np.array(action)
                a = a.reshape(a.shape[0],1)
                q = self.critic_model.model([tf.convert_to_tensor(tf_prev_state), tf.convert_to_tensor(a)]).numpy()

                self.tm.add_transition(prev_state,action,reward,state,q,done)
                episodic_reward += reward       

                state_batch, action_batch, reward_batch,next_state_batch,transitions=self.comput_loss_and_update2()
                self.update_target(self.target_actor.model.variables, self.actor_model.model.variables, self.tau)
                self.update_target(self.target_critic.model.variables, self.critic_model.model.variables, self.tau)                
                #if(not first_qt_trained):
                                       
                                
                if (count % trainQTFreqquency == 0 and count > learningStartIter): 
                    #print("train qt---->",count)
                    first_qt_trained=True
                    self.qt.train_q_prediction_withou_validation(n_epochs=lstm_epochs)
                
                if(first_qt_trained and (count % update_QTCritic_frequency == 0) and (count > learningStartIter)):
                        #print("update critic--->",count)
                        self.update_critic_target(state_batch, action_batch, reward_batch,next_state_batch,transitions)
                if(count % 10000 == 0):
                    self.evaluate(trial,count)

                count+=1

                # End this episode when `done` is True
                if done:            
                    run=False
                prev_state = state

                if(done):
                    avg_trial_rew = avg_reward = np.mean(ep_reward_list[-11:]) if len(ep_reward_list)>0 else 0            
                    log_itr+=1
                    now = datetime.now()        
                    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
                    log_data_dict =[
                    ('Count',log_itr),
                    ('Task',self.task_name),
                    ('Time',dt_string),
                    ('TMCount',self.tm.__len__()),
                    ('RTMCount',self.rtm.__len__()),
                    ('Ep', ep),
                    ('Itr', itr),
                    ("Done",done),
                    ('Rew', episodic_reward),
                    ('AvgEpRew', (episodic_reward/itr)),
                    ('AvgLastEp', avg_trial_rew)]
                    self.log(log_data_dict) 

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)


def config_trial_logger(base_log_dir = "./log/trials/"):    
        tl.session(base_log_dir).__enter__()

def trial_log(log_data_dict):
        for k, v in log_data_dict:
            tl.logkv(k, v)
        tl.dumpkvs()

def grid_search():
    total_episodes=[200]
    lstm_epochs=[15]
    learningStartIter=[1]    
    trainQTFreqquency=[4]    
    update_QTCritic_frequency=[1]
    q_lstm_bsize=[10000]    
    trial=2
    config_trial_logger()

    for tep in total_episodes:
        for lstmep in lstm_epochs:
            for tqt in trainQTFreqquency:
                for start in learningStartIter:
                    for upcritic in update_QTCritic_frequency:
                        for bs in q_lstm_bsize:
                            trial+=1
                            now = datetime.now()
                            dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
                            log_data_dict =[('Trial',trial),('Time',dt_string),('TotalEp',tep),
                            ('Tqt',tqt),('Lstmep',lstmep),('StartLearn',start),('Upcritic',upcritic),
                            ('Qlstm_bs',bs)]
                            trial_log(log_data_dict)
                            agent = COMPERDDPG()
                            agent.train(
                                total_episodes=tep,
                                lstm_epochs=lstmep,
                                trainQTFreqquency=tqt,
                                learningStartIter= start,
                                update_QTCritic_frequency=upcritic,
                                q_lstm_bsize=bs,
                                trial=trial)                        

def main():
    grid_search()

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(type(e),e) 




            








