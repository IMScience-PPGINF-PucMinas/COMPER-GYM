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
from datetime import datetime
import os
from tensorflow.keras import backend as K


tf.enable_eager_execution()


task_name = "Pendulum-v1"
env = GymEnv(task_name)


base_log_dir = "./log_comper_ddpg/train/"
logger.session(base_log_dir).__enter__()

def log(log_data_dict):
    for k, v in log_data_dict:
        logger.logkv(k, v)
    logger.dumpkvs()


def config_memories():               
    tm = TM(max_size=100000,name="tm", memory_dir="./")
    rtm  = RTM(max_size=100000,name="rtm",memory_dir="/.")
    return tm,rtm    

# Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
# TensorFlow to build a static graph out of the logic and computations in our function.
# This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
@tf.function
def update_actor_critic_nets(state_batch, action_batch, reward_batch, next_state_batch,target_predicted):
    # Training and updating Actor & Critic networks.
    # See Pseudo Code.    
    with tf.GradientTape() as tape:
        #target_actions = target_actor.model(next_state_batch, training=True)
        #y = reward_batch + gamma * target_critic.model([next_state_batch, target_actions], training=True)        
        y = reward_batch + gamma * qt.lstm.lstm(target_predicted, training=True)
        #y = reward_batch + gamma * target_predicted

        critic_value = critic_model.model([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, critic_model.model.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, critic_model.model.trainable_variables)
    )

    with tf.GradientTape() as tape:
        actions = actor_model.model(state_batch, training=True)
        critic_value = critic_model.model([state_batch, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, actor_model.model.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, actor_model.model.trainable_variables)
    )

def _get_transition_components(transitions):
        st_1 = transitions[:,:ft.T_IDX_ST_1[1]]
        a = transitions[:,ft.T_IDX_A]
        r = transitions[:,ft.T_IDX_R] # get rewards
        st = transitions[:,ft.T_IDX_ST[0]:ft.T_IDX_ST[1]]
        q = transitions[:,ft.T_IDX_Q]# To ilustrate, but we do not need this here.
        done = transitions[:,ft.T_IDX_DONE] # get done signals
        return st_1,a,r,st,q,done

# We compute the loss and update parameters
def comput_loss_and_update():
    transitions_batch=[]   
    if(rtm. __len__()>0):
        transitions_batch = rtm.sample_transitions_batch(64)
    else:
        transitions_batch = tm.sample_transitions_batch(64)
    
    st_1,a,r,st,q,done = _get_transition_components(transitions_batch)
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
    update_actor_critic_nets(state_batch, action_batch, reward_batch,next_state_batch,transitions)

    critic_value = critic_model.model([state_batch, action_batch], training=True).numpy()
    for i in range(len(st_1)):                 
        tm.add_transition(st_1[i],a[i],r[i],st[i],critic_value[i],float(done[i]))

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


actor_model = actor_critic.get_actor(env.num_states,env.upper_bound)
critic_model = actor_critic.get_critic(env.num_states,env.num_actions)

target_actor = actor_critic.get_actor(env.num_states,env.upper_bound)
#target_critic = actor_critic.get_critic(env.num_states,env.num_actions)

# Making the weights equal initially
target_actor.model.set_weights(actor_model.model.get_weights())
#target_critic.model.set_weights(critic_model.model.get_weights())

tm, rtm = config_memories()
transitin_size = int((2*env.num_states + env.num_actions + 1))
qt = QLSTMGSCALE(transitions_memory=tm,reduced_transitions_memory=rtm,
                              inputshapex=1,inputshapey=transitin_size,outputdim=env.num_actions,
                              verbose=False,transition_batch_size=64,netparamsdir='dev',
                              target_optimizer="rmsprop",log_dir='dev')


# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []




# Takes about 4 min to train
log_itr=0
logFrequency=100
trainQTFreqquency=100    
learningStartIter=1
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0
    itr = 1   
    run =True
    while run:
        itr+=1
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy.get_action(tf_prev_state,actor_model.model,env.lower_bound,env.upper_bound)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)      
        a = np.array(action)
        a = a.reshape(a.shape[0],1)
        q = critic_model.model([tf.convert_to_tensor(tf_prev_state), tf.convert_to_tensor(a)]).numpy()

        tm.add_transition(prev_state,action,reward,state,q,done)
        episodic_reward += reward       

        comput_loss_and_update()
        update_target(target_actor.model.variables, actor_model.model.variables, tau)
        #update_target(target_critic.model.variables, critic_model.model.variables, tau)
        if (itr % trainQTFreqquency == 0 and itr > learningStartIter):
            qt.train_q_prediction(n_epochs=5)
            
        # End this episode when `done` is True
        if done:
            run=False
        prev_state = state

        if(((itr+1) % logFrequency == 0) or done):
            avg_trial_rew = np.mean(ep_reward_list) if len(ep_reward_list)>0 else 0            
            log_itr+=1
            now = datetime.now()        
            dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
            log_data_dict =[
            ('Count',log_itr),
            ('Task',task_name),
            ('Time',dt_string),
            ('Ep', ep),
            ('Itr', itr),
            ('Rew', episodic_reward),
            ('AvgEpRew', (episodic_reward/itr)),
            ('AvgTrialRew', np.mean(avg_trial_rew))]
            log(log_data_dict) 

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)








# Plotting graph
# Episodes versus Avg. Rewards
#plt.plot(avg_reward_list)
#plt.xlabel("Episode")
#plt.ylabel("Avg. Epsiodic Reward")
#plt.show()
