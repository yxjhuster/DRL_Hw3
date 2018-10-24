import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, sess, model, lr, num_episodes, render, env):
        self.model = model

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here. 
        self.gamma = 0.99
        self.lr = lr
        self.num_episodes = num_episodes
        self.env = env
        self.scale_factor = 1e-2
        self.render = render
        self.sess = sess

        self.log_layer = tf.log(self.model.output)
        self.g_value = tf.placeholder(tf.float32)
        self.action_input = tf.placeholder(tf.int64)
        self.step_num = tf.placeholder(tf.float32)
        self.depth = 4

        self.action_one_hot = tf.one_hot(self.action_input, self.depth)
        self.loss = -tf.divide(tf.reduce_sum(self.g_value * \
                                (self.log_layer * self.action_one_hot)), self.step_num)
        self.train_op = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)
        initializer = tf.global_variables_initializer()
        writer = tf.summary.FileWriter('./graphs', self.sess.graph)
        self.sess.run(initializer)

        self.num_test_episode = 100
        self.num_ploting_episode = 1000
        self.mean_list = []
        self.std_list = []


    def adjust_obs_format(self, observation):
        return observation.reshape(1, self.env.observation_space.shape[0])


    def train(self):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        for idx_episode in range(self.num_episodes):
            states, actions, rewards = self.generate_episode()

            total_step_num = len(states)
            G = []
            # get the discounted reward
            for time in range(total_step_num):
                Gt = 0
                for k in range(time, total_step_num):
                    Gt += self.gamma**(k-time) * (rewards[k])
                G.append(Gt)
            G_mean, G_std = np.mean(G), np.std(G)
            G_reduced = (G-G_mean) / G_std
            G_reduced = np.array(G_reduced).reshape([total_step_num, 1])
            G_reduced = G_reduced * self.scale_factor # G[0] is the discounted reward


            prob_layer = self.sess.run(self.model.output, 
                                        feed_dict = {self.model.input: np.vstack(states)})

            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.model.input: np.vstack(states),
                self.action_input: np.array(actions),
                self.g_value: G_reduced,
                self.step_num: float(total_step_num)
                })

            # print('Episode %d, Reward %.1f, Loss %.7f' % (idx_episode, G[0], loss))
            print('Episode %d, Reward %.1f, Loss %.7f' % (idx_episode, np.sum(rewards), loss))

            if idx_episode % self.num_ploting_episode == 0:
                self.test()

        self.plot()
        return


    def test(self):
        cumulative_reward = []
        for idx in range(self.num_test_episode):
            states, actions, rewards = self.generate_episode()

            total_step_num = len(states)
            G = []
            # get the discounted reward
            for time in range(total_step_num):
                Gt = 0
                for k in range(time, total_step_num):
                    Gt += self.gamma**(k-time) * (rewards[k])
                G.append(Gt)

            # cumulative_reward.append(G[0])
            cumulative_reward.append(np.sum(rewards))
        reward_mean = np.mean(np.array(cumulative_reward))
        self.mean_list.append(reward_mean)
        reward_std = np.std(np.array(cumulative_reward))
        self.std_list.append(reward_std)


    def plot(self):
        fig, (axes) = plt.subplots(nrows=1)
        x = np.array(range(0, self.num_episodes, self.num_ploting_episode))
        y = np.array(self.mean_list)
        stds = np.array(self.std_list)
        axes.errorbar(x, y, yerr=stds, fmt='-o')
        plt.savefig('test_reinforce.png')


    def generate_episode(self):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []

        observation = self.env.reset()
        while True:
            if self.render == True:
                self.env.render()
            states.append(observation)
            action_distribution = self.sess.run(self.model.output, feed_dict={
                                                self.model.input: self.adjust_obs_format(observation)})
            action = np.random.choice(len(action_distribution[0]), 
                                        p=action_distribution[0])
            actions.append(action)
            next_observation, reward, done, info = self.env.step(action)  
            rewards.append(reward)

            observation = next_observation
            if done: break

        self.env.close()   
        return states, actions, rewards


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
    lr = 3*0.0001
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())
    f.close()

    # TODO: Train the model using REINFORCE and plot the learning curve.
    reinforce = Reinforce(sess, model=model, lr=lr, num_episodes=num_episodes, render=render, env=env)
    reinforce.train()


if __name__ == '__main__':
    main(sys.argv)
