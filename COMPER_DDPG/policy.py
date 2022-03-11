import tensorflow as tf
import numpy as np

def get_action(state,actor_model,lower_bound, upper_bound,noise_object):    
    sampled_actions=tf.squeeze(actor_model(state))
    noise = noise_object()
    sampled_actions = sampled_actions + noise
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]

def get_action_no_noise(state,actor_model,lower_bound, upper_bound):
    sampled_actions=tf.squeeze(actor_model(state))
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]

class Epsilon(object):
    def __init__(self, schedule_timesteps, final_p, initial_p):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):        
        fraction = min(1.0, float(t) / self.schedule_timesteps)
        return self.initial_p + (self.final_p - self.initial_p) * fraction
        #return self.initial_p + (self.initial_p+self.final_p) * fraction