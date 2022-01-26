from data import evaluate_logger as elog
from environment.env import GymEnv
import tensorflow as tf
import policy


elog.session("./log_comper_ddpg/evaluate/").__enter__()

def evaluate_agente(actor_model,trial,iterations,n_episodes=10,task_name = "Pendulum-v1"):
    env = GymEnv(task_name)
    ep_reward_list = []

    for ep in range(n_episodes):
        ep_reward_list = []
        prev_state = env.reset()
        episodic_reward = 0
       
        while not done:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = policy.get_action_no_noise(tf_prev_state,actor_model.model,env.lower_bound,env.upper_bound)
            state, reward, done, info = env.step(action)
            episodic_reward += reward
            prev_state = state
            ep_reward_list.append(episodic_reward)
    
    avg_trial_rew = np.mean(ep_reward_list) if len(ep_reward_list)>0 else 0            
    log_itr+=1
    now = datetime.now()        
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    log_data_dict =[
    ('Trial',trial),
    ('Time',dt_string),
    ('Task',self.task_name),
    ('Int',iterations),
    ('EvalAvgRew', avg_trial_rew)]
    for k, v in log_data_dict:
        elog.logkv(k, v)
        elog.dumpkvs()

        