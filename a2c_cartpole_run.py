# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 20:51:17 2020

@author: muazi
"""
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
tf.reset_default_graph()
sess = tf.Session() # Set up networks 
actor = actor(sess, env)
critic = critic(sess, env)
init = tf.global_variables_initializer()
sess.run(init)
rewards = one_step_a2c(env, actor, critic, gamma=0.9) 
# Smooth rewards and error with moving average 
window = 10
rewards_smooth = [np.mean(rewards[i:i+window]) 
    if i > window 
    else np.mean(rewards[0:i+1]) 
    for i in range(len(rewards))] 
 
# Plot results 
plt.figure(figsize=(12,8))
plt.plot(rewards, label='Rewards')
plt.plot(rewards_smooth, label='Smoothed Rewards')
plt.title('Total Rewards')
plt.legend(loc='best')
plt.ylabel('Rewards')
plt.show()