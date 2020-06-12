# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:35:58 2020

@author: muazi
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
%matplotlib inline

def one_step_a2c(env, actor, critic, gamma=0.99, 
                 num_episodes=2000, print_output=True):
    '''
    Inputs
    =====================================================
    env: class, OpenAI environment such as CartPole
    actor: class, parameterized policy network
    critic: class, parameterized value network
    gamma: float, 0 < gamma <= 1, determines the discount
            factor to be applied to future rewards.
    num_episodes: int, the number of episodes to be run.
    print_output: bool, prints algorithm settings and 
            average of last 10 episodes to track training.
     
    Outputs
    ======================================================
    ep_rewards: np.array, sum of rewards for each 
            simulated episode
    '''
     
    # Set up vectors for episode data
    ep_rewards = np.zeros(num_episodes)
     
    action_space = np.arange(env.action_space.n)
     
    for ep in range(num_episodes):
         
        s_0 = env.reset()
        complete = False
        actions = []
        rewards = []
        states = []
        targets = []
        errors = []
        t = 0
         
        while complete == False:
             
            # Select and take action
            action_probs = actor.predict(s_0.reshape(1, -1))
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)
             
            # Calculate predictions and error
            predicted_value = critic.predict(s_1.reshape(1, -1))
            target = r + gamma * predicted_value
            error = target - critic.predict(s_0.reshape(1, -1))
                 
            # Log results
            states.append(s_0)
            actions.append(action)
            rewards.append(r)
            targets.append(target)
            errors.append(error)
             
            # Update networks
            actor.update(states=np.array(states),
                         actions=np.array(actions),
                         returns=np.array(errors))
            critic.update(states=np.array(states),
                          returns=np.array(targets))
             
            t += 1
            s_0 = s_1
             
            if complete:
                ep_rewards[ep] = np.sum(rewards)
                 
                # Print average of last 10 episodes if true
                if print_output and (ep + 1) % 10 == 0 and ep != 1:
                    avg_rewards = np.mean(ep_rewards[ep-10:ep+1])
                    print("\rOne-step A2C at Episode: {:d}, Avg Reward: {:.2f}".format(
                            ep + 1, avg_rewards), end="")
                 
    return ep_rewards

class policy_estimator(object):
    
    def __init__(self, sess, env):
        # Pass TensorFlow session object
        self.sess = sess
        # Get number of inputs and outputs from environment
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.learning_rate = 0.01
        
        # Define number of hidden nodes
        self.n_hidden_nodes = 16
        
        # Set graph scope name
        self.scope = "policy_estimator"
        
        # Create network
        with tf.variable_scope(self.scope):
            initializer = tf.contrib.layers.xavier_initializer()
            
            # Define placholder tensors for state, actions, and rewards
            self.state = tf.placeholder(tf.float32, [None, self.n_inputs], 
                                        name='state')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
            self.actions = tf.placeholder(tf.int32, [None], name='actions')
            
            layer_1 = fully_connected(self.state, self.n_hidden_nodes,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)
            output_layer = fully_connected(layer_1, self.n_outputs,
                                           activation_fn=None,
                                           weights_initializer=initializer)
            
            # Get probability of each action
            self.action_probs = tf.squeeze(
                tf.nn.softmax(output_layer - tf.reduce_max(output_layer)))
            
            # Get indices of actions
            indices = tf.range(0, tf.shape(output_layer)[0]) \
                * tf.shape(output_layer)[1] + self.actions
                
            selected_action_prob = tf.gather(tf.reshape(self.action_probs, [-1]),
                                             indices)
    
            # Define loss function
            self.loss = -tf.reduce_mean(tf.log(selected_action_prob) * self.rewards)

            # Get gradients and variables
            self.tvars = tf.trainable_variables(self.scope)
            self.gradient_holder = []
            for j, var in enumerate(self.tvars):
                self.gradient_holder.append(tf.placeholder(tf.float32, 
                    name='grads' + str(j)))
            
            self.gradients = tf.gradients(self.loss, self.tvars)
            
            # Minimize training error
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.gradient_holder, self.tvars))
            
    def predict(self, state):
        probs = self.sess.run([self.action_probs], 
                              feed_dict={
                                  self.state: state
                              })[0]
        return probs
    
    def update(self, gradient_buffer):
        feed = dict(zip(self.gradient_holder, gradient_buffer))
        self.sess.run([self.train_op], feed_dict=feed)

    def get_vars(self):
        net_vars = self.sess.run(tf.trainable_variables(self.scope))
        return net_vars

    def get_grads(self, states, actions, rewards):
        grads = self.sess.run([self.gradients], 
            feed_dict={
            self.state: states,
            self.actions: actions,
            self.rewards: rewards
            })[0]
        return grads   


class value_estimator(object):
    
    def __init__(self, sess, env):
        # Pass TensorFlow session object
        self.sess = sess
        # Get number of inputs and outputs from environment
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = 1
        self.learning_rate = 0.01
        
        # Define number of hidden nodes
        self.n_hidden_nodes = 16
        
        # Set graph scope name
        self.scope = "value_estimator"
        
        # Create network
        with tf.variable_scope(self.scope):
            initializer = tf.contrib.layers.xavier_initializer()
            
            # Define placholder tensors for state, actions, and rewards
            self.state = tf.placeholder(tf.float32, [None, self.n_inputs], 
                                        name='state')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
            
            layer_1 = fully_connected(self.state, self.n_hidden_nodes,
                                      activation_fn=tf.nn.relu,
                                      weights_initializer=initializer)
            output_layer = fully_connected(layer_1, self.n_outputs,
                                           activation_fn=None,
                                           weights_initializer=initializer)
            
            self.state_value_estimation = tf.squeeze(output_layer)
    
            # Define loss function as squared difference between estimate and 
            # actual
            self.loss = tf.reduce_mean(tf.squared_difference(
                self.state_value_estimation, self.rewards))

            # Get gradients and variables
            self.tvars = tf.trainable_variables(self.scope)
            self.gradient_holder = []
            for j, var in enumerate(self.tvars):
                self.gradient_holder.append(tf.placeholder(tf.float32, 
                    name='grads' + str(j)))
            
            self.gradients = tf.gradients(self.loss, self.tvars)
            
            # Minimize training error
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.apply_gradients(
                zip(self.gradient_holder, self.tvars))
            
    def predict(self, state):
        value_est = self.sess.run([self.state_value_estimation], 
                              feed_dict={
                                  self.state: state
                              })[0]
        return value_est
    
    def update(self, gradient_buffer):
        feed = dict(zip(self.gradient_holder, gradient_buffer))
        self.sess.run([self.train_op], feed_dict=feed)

    def get_vars(self):
        net_vars = self.sess.run(tf.trainable_variables(self.scope))
        return net_vars

    def get_grads(self, states, rewards):
        grads = self.sess.run([self.gradients], 
            feed_dict={
            self.state: states,
            self.rewards: rewards
            })[0]
        return grads   

env = gym.make('CartPole-v0')
tf.reset_default_graph()
sess = tf.Session() # Set up networks 
actor = networks.actor(sess, env)
critic = networks.critic(sess, env)
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