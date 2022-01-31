import gym
from sqlalchemy import false
import tensorflow as tf
import numpy as np
from config import transitions
import actor_critic
import policy
from memory.tm.transitions_memory import TransitionsMemory as TM
from memory.rtm.reduced_transitions_memory import ReducedTransitionsMemory as RTM
from config.transitions import FrameTransition as ft 
from qrnn.q_lstm_gscale import QLSTMGSCALE
from environment.env import GymEnv
import data_log as dlog
from datetime import datetime

import os
class COMPERDDPG(object):
    def __init__(self,task_name = "Pendulum-v1") -> None:
        super().__init__()
        self.task_name = task_name
        self.env = GymEnv(task_name)
        self.tm = object
        self.rtm = object
        self.critic_model = object
        self.qt = object
        self.actor_model = object
        self.target_actor = object
        self.target_critic = object
        self.critic_lr = 0.001
        self.actor_lr = 0.001
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)             
        self.gamma = 0.99       
        self.tau = 0.001
    

    def config_memories(self):               
        self.tm = TM(max_size=1000000,name="tm", memory_dir="./")
        self.rtm  = RTM(max_size=1000000,name="rtm",memory_dir="/.")
            

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
    
    def get_transitions_batch(self):
        transitions_batch=[]   
        if(self.rtm. __len__()>0):
            transitions_batch = self.rtm.sample_transitions_batch(100)
        else:
            transitions_batch = self.tm.sample_transitions_batch(100)        
        return transitions_batch

    def comput_loss_and_update2(self):
        transitions_batch=[]   
        if(self.rtm. __len__()>0):
            transitions_batch = self.rtm.sample_transitions_batch(256)
        else:
            transitions_batch = self.tm.sample_transitions_batch(256)
        
        st_1,a,r,st,q,done = self._get_transition_components(transitions_batch)
        a = np.array(a)
        #a = a.reshape(a.shape[0],1)
        r = np.array(r)
        #r = r.reshape(r.shape[0],1)
        state_batch = tf.convert_to_tensor(st_1)
        action_batch = tf.convert_to_tensor(a)
        reward_batch = tf.convert_to_tensor(r)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(st)

        self.update_actor_critic_nets(state_batch, action_batch, reward_batch,next_state_batch)

        transitions = transitions_batch[:,:-2]
        critic_value = self.critic_model.model([state_batch, action_batch], training=True).numpy()

        for i in range(len(st_1)):                 
            self.tm.add_transition(st_1[i],a[i],r[i],st[i],critic_value[i],float(done[i]))
        
        return state_batch, action_batch, reward_batch,next_state_batch,transitions
    
    def update_critic_target(self,state_batch, action_batch, reward_batch, next_state_batch,target_predicted):
        with tf.GradientTape() as tape:
            y =self.qt.predict(target_predicted)
            critic_value = self.critic_model.model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.model.trainable_variables))

    def update_target(self,target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def log_evaluation(self,trial,dt_string,task_name,iterations,avg_trial_rew):
        now = datetime.now()        
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        dlog.log_eval_data(trial,dt_string,task_name,iterations,avg_trial_rew)

    def evaluate(self,trial,iterations,n_episodes=50):
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
        self.log_evaluation(trial,dt_string,self.task_name,iterations,avg_trial_rew)


    def log_train(self,trial,ep,log_itr,ep_itr,done,episodic_reward,ep_reward_list):
        avg_trial_rew = np.mean(ep_reward_list) if len(ep_reward_list)>0 else 0             
        last_ep_avg_reward = np.mean(ep_reward_list[-10:])
        now = datetime.now()        
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        dlog.log_train_data(trial,log_itr,self.task_name,dt_string,self.tm.__len__(),self.rtm.__len__(),
                            ep,ep_itr,done,episodic_reward,(episodic_reward/ep_itr),last_ep_avg_reward, avg_trial_rew) 
        avg_reward = np.mean(ep_reward_list[-10:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))

    def train(self,total_steps=100,lstm_epochs=150,update_QTCritic_frequency=5,trainQTFreqquency=100,learningStartIter=1,q_lstm_bsize=1000,trial=1,
                evaluate_frequency = 5000,evaluate_epsodes = 50):
          
        qlstm_log_path = "./log/train/qlstm_"+"trial"+str(trial)+"/" 
                
        self.actor_model = actor_critic.get_actor(self.env.num_states,self.env.upper_bound,self.env.num_actions)
        self.critic_model = actor_critic.get_critic(self.env.num_states,self.env.num_actions)

        self.target_actor = actor_critic.get_actor(self.env.num_states,self.env.upper_bound,self.env.num_actions)
        self.target_critic = actor_critic.get_critic(self.env.num_states,self.env.num_actions)
        
        self.target_actor.model.set_weights(self.actor_model.model.get_weights())
        self.target_critic.model.set_weights(self.critic_model.model.get_weights())

    def create_q_target(self,q_lstm_bsize,qlstm_log_path):
        transitin_size=ft.T_LENGTH -2
        self.qt = QLSTMGSCALE(
                    transitions_memory=self.tm,
                    reduced_transitions_memory=self.rtm,
                    inputshapex=1,
                    inputshapey=transitin_size,
                    outputdim=self.env.num_actions,
                    verbose=False,
                    transition_batch_size=q_lstm_bsize,
                    netparamsdir='dev',
                    target_optimizer="rmsprop",
                    log_dir=qlstm_log_path,
                    target_early_stopping=True)

    def train(self,total_steps=100000,lstm_epochs=150,update_QTCritic_frequency=5,trainQTFreqquency=100,learningStartIter=1,q_lstm_bsize=1000,trial=1,
                evaluate_frequency = 5000,evaluate_epsodes = 50):
        logFrequency=100
        base_log_dir="./log/train/"
        rew_log_path = base_log_dir+"trial"+str(trial)+"/"
        qlstm_log_path = rew_log_path+"lstm/"
        
        self.create_actor_critic()
        self.config_memories()
        self.create_q_target(q_lstm_bsize,qlstm_log_path)
        
        ep_reward_list = []      
        log_itr=0
        
        first_qt_trained = False
        ep=1
        ep_itr = 0
        episodic_reward = 0
        prev_state = self.env.reset()

        for step in range(total_steps):
            ep_itr+=1
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = policy.get_action(tf_prev_state,self.actor_model.model,self.env.lower_bound,self.env.upper_bound)            
            state, reward, done, info = self.env.step(action)      
                
            q = self.critic_model.model([tf.convert_to_tensor(tf_prev_state), tf.convert_to_tensor(action)]).numpy()
            action = np.array(action)
            action = action.reshape(action.shape[1])
            self.tm.add_transition(prev_state,action,reward,state,q,done)
            episodic_reward += reward       

            state_batch, action_batch, reward_batch,next_state_batch,transitions=self.comput_loss_and_update2()
            self.update_target(self.target_actor.model.variables, self.actor_model.model.variables, self.tau)
            self.update_target(self.target_critic.model.variables, self.critic_model.model.variables, self.tau)                
                                
            if(step % trainQTFreqquency == 0 and step > learningStartIter): 
                print("train qt---->",step)
                first_qt_trained=True                                     
                self.qt.train_q_prediction_withou_validation(n_epochs=lstm_epochs)
                
            if(first_qt_trained and (step % update_QTCritic_frequency == 0) and (step > learningStartIter)):
                print("update critic--->",step)
                self.update_critic_target(state_batch, action_batch, reward_batch,next_state_batch,transitions)
            
            if(step % evaluate_frequency==0 and step > learningStartIter):
                print("evaluate agent--->",step)
                self.evaluate(trial,step,evaluate_epsodes)

            prev_state = state

            if done:
                prev_state = self.env.reset()
                ep_reward_list.append(episodic_reward)                
                log_itr+=1                
                self.log_train(trial,ep,log_itr,ep_itr,done,episodic_reward,ep_reward_list)
                ep+=1
                ep_itr=0
                episodic_reward = 0
                done=false

            #if((step % logFrequency == 0) or done):
             #  log_itr+=1
              # self.log_train(trial,ep,log_itr,itr,done,episodic_reward,ep_reward_list)

            if(done):
                ep_reward_list.append(episodic_reward)
                log_itr+=1
                self.log_train(trial,ep,log_itr,itr,done,episodic_reward,ep_reward_list)
                prev_state = self.env.reset()                                
                episodic_reward = 0
                avg_reward = np.mean(ep_reward_list[-40:])
                print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
                itr = 0
                ep+=1

           
            
            
def grid_search():
    total_episodes=[3000000]
    lstm_epochs=[500]
    learningStartIter=[1]    
    trainQTFreqquency=[100]    
    update_QTCritic_frequency=[100]
    q_lstm_bsize=[10000]    
    trial=0
    evaluate_frequency = 5000
    evaluate_epsodes = 50
   
    for tep in total_episodes:
        for lstmep in lstm_epochs:
            for tqt in trainQTFreqquency:
                for start in learningStartIter:
                    for upcritic in update_QTCritic_frequency:
                        for bs in q_lstm_bsize:
                            trial+=1
                            now = datetime.now()
                            dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
                            dlog.log_trial_data(trial,dt_string,tep,tqt,lstmep,start,upcritic,bs)
                            agent = COMPERDDPG(task_name="Hopper-v2")
                            agent.train(
                                total_steps=tep,
                                lstm_epochs=lstmep,
                                trainQTFreqquency=tqt,
                                learningStartIter= start,
                                update_QTCritic_frequency=upcritic,
                                q_lstm_bsize=bs,
                                trial=trial,
                                evaluate_frequency=evaluate_frequency,
                                evaluate_epsodes=evaluate_epsodes)                        

def test_gym():
    env = gym.make('Hopper-v2')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            #env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

def main():
    #test_gym()
    grid_search()

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(type(e),e) 




            








