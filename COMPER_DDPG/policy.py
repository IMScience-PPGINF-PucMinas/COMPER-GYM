import tensorflow as tf
import numpy as np
from action_exploration import OUActionNoise


def get_action(state,actor_model,lower_bound, upper_bound):
    std_dev = 0.2
    noise_object = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
    #sampled_actions = actor_model.forward(state)
    sampled_actions=tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]