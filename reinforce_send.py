#!/usr/bin/env python
import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.optimizers import Adam

import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, sess, model, lr, num_episodes, render, env):
        self.env, self.render = env, render
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  
        self.gamma = 0.99  # 1.0
        self.learning_rate = 3*0.0001  #1e-05  # 0.0001
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.num_episodes = num_episodes
        self.reward_factor = 1e-02
        self.sess = sess

        self.policy_model = model  # This is the policy model, pi(A|S, theta)
        self.log_prob_layer = tf.log(self.policy_model.output)

        self.return_holder = tf.placeholder(dtype=tf.float32)  # Placeholder for a list of returns
        self.action_idx_holder = tf.placeholder(dtype=tf.int32)  # Placeholder for a given action
        self.depth = 4
        self.action_one_hot = tf.one_hot(self.action_idx_holder, self.depth)  # one_hot vector for actions

        # Maximize logP * V = Minimize -logP * V
        self.loss_divider = tf.placeholder(dtype=tf.float32)
        self.loss = -tf.divide(tf.reduce_sum(self.return_holder * \
                                (self.log_prob_layer * self.action_one_hot)), self.loss_divider)
        self.opt_operation = self.optimizer.minimize(self.loss)

        initializer = tf.global_variables_initializer()
        writer = tf.summary.FileWriter('./graphs', self.sess.graph)
        self.sess.run(initializer)

        self.episode_counter = 0
        self.num_test_episodes = 100
        self.num_episodes_plotting = 1000
        self.mean_reward_list, self.std_reward_list = [], []

    def adjust_obs_format(self, observation):
        return observation.reshape(1, self.env.observation_space.shape[0])

    def train(self):
        # Trains the model on a single episode using REINFORCE
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data

        for idx_episode in range(self.num_episodes):
            self.states, self.actions, self.action_probs, self.rewards, reward_episode = self.generate_episode()

            T = len(self.states)
            Gs = []
            # Get the return from step t to T
            for time in range(T):
                Gt = 0
                for k in range(time, T):
                    Gt += self.gamma**(k-time) * (self.rewards[k])   # *self.reward_factor
                Gs.append(Gt)
            # print(Gs)
            mean, std = np.mean(Gs), np.std(Gs)
            Gs = (Gs-mean) / std
            Gs = np.array(Gs).reshape([T, 1])
            
            prob_layer = self.sess.run(self.policy_model.output, feed_dict={
                         self.policy_model.input: np.vstack(self.states)})
            
            loss_episode, _ = self.sess.run([self.loss, self.opt_operation], feed_dict={
                self.policy_model.input: np.vstack(self.states),
                self.action_idx_holder: np.array(self.actions),
                self.return_holder: Gs,
                self.loss_divider: float(T)
                })

            print('Episode %d, Reward %.1f, Loss %.7f' % (idx_episode, reward_episode, loss_episode))

            if self.episode_counter % self.num_episodes_plotting == 0:
                self.test_in_training_process()

            self.episode_counter += 1

        self.plot()

    def get_action(self, observation):
        action_probility_distribution = self.policy_model.predict(self.adjust_obs_format(observation))
        action = np.random.choice(len(action_probility_distribution[0]), 
                                  p=action_probility_distribution[0])
        return action, action_probility_distribution

    def test_in_training_process(self):
        test_reward_episode_list = []
        for idx_test in range(self.num_test_episodes):
            self.env.reset()
            observation = self.env.reset()
            reward_episode = 0
            while True:
                action, action_probility_distribution = self.get_action(observation)

                new_observation, reward, done, info = self.env.step(action)
                reward_episode += reward
                observation = new_observation

                if done: break
            test_reward_episode_list.append(reward_episode)
            self.env.close()

        reward_mean = np.mean(np.array(test_reward_episode_list))
        self.mean_reward_list.append(reward_mean)
        reward_std = np.std(np.array(test_reward_episode_list))
        self.std_reward_list.append(reward_std)

    def plot(self):
        fig, (axes) = plt.subplots(nrows=1)
        # fig = plt.figure(0)
        x = np.array(range(0, self.num_episodes, self.num_episodes_plotting))
        y = np.array(self.mean_reward_list)
        stds = np.array(self.std_reward_list)
        axes.errorbar(x, y, yerr=stds, fmt='-o')
        # plt.errorbar(x, y, stds_plot, fmt='-o')
        # plt.show()
        plt.savefig('test.png')

    def generate_episode(self):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states, actions, action_probs, rewards, reward_episode = [], [], [], [], 0
        observation = self.env.reset()

        while True:
            # Get an action according to the current policy model
            # if self.render: self.env.render()
            # self.env.render()
                
            states.append(observation)
            # Network input - state
            # Network output - softmax action probility
            action, action_probility_distribution = self.get_action(observation)  # A0
            actions.append(action)
            action_probs.append(action_probility_distribution)

            new_observation, reward, done, info = self.env.step(action)
            
            rewards.append(reward)
            reward_episode += reward

            observation = new_observation
            if done: break

        self.env.close()
        return states, actions, action_probs, rewards, reward_episode



def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)
    # Setting this as the default tensorflow session. 
    keras.backend.tensorflow_backend.set_session(sess)

    # Parse command-line arguments.
    args = parse_arguments()
    
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the policy model from file.
    with open(model_config_path, 'r') as file_model_config:
        model = keras.models.model_from_json(file_model_config.read())
    file_model_config.close()

    # TODO: Train the model using REINFORCE and plot the learning curve.
    reinforce_agent = Reinforce(sess, model=model, lr=lr, num_episodes=num_episodes, render=render, env=env)
    reinforce_agent.train()


if __name__ == '__main__':
    main(sys.argv)
