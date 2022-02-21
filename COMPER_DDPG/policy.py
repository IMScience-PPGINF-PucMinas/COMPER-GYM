import tensorflow as tf
import numpy as np



def get_action(state,actor_model,lower_bound, upper_bound,noise_object):    
    sampled_actions=tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

def get_action_no_noise(state,actor_model,lower_bound, upper_bound):
    sampled_actions=tf.squeeze(actor_model(state))
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]