import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr, env, sess, render = False):
        self.model = model

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here. 
        self.env = env
        self.sess = sess
        self.observation_space_size = 8
        self.action_space_size = 4
        self.action_space = range(self.action_space_size)
        self.render = render
        self.down_scale_factor = 1e-2
        # states, actions, rewards = self.generate_episode(self.render)
        self.actions_input = tf.placeholder(tf.int64)
        # self.rewards_input = tf.placeholder(tf.float32)
        self.total_steps = tf.placeholder(tf.float32)
        self.G_value = tf.placeholder(tf.float32)
        # log_value = self.get_log_value(self.model.input, self.actions_input)
        # self.G = self.get_g_value(self.rewards_input, self.total_steps, gamma)
        self.depth = 4
        self.a_matrix = tf.one_hot(self.actions_input, self.depth)
        # print(tf.shape(self.a_matrix))
        self.log_value = tf.reduce_sum(tf.log(self.model.output) * self.a_matrix, 1)
        # print(tf.shape(self.log_value))
        # self.loss = tf.reduce_sum(self.G_value * self.log_value)/tf.cast(self.total_steps, tf.float32)
        self.loss = -tf.divide(tf.reduce_sum(self.G_value *  self.log_value), self.total_steps)
        # print(tf.shape(self.loss))
        # print(loss)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)


    def train(self, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, actions, rewards = self.generate_episode(self.render)
        # print(actions)
        total_step_num =  np.shape(states)[0]
        # log_value = self.get_log_value(sess, states, actions)
        cumulative_reward = self.get_g_value(rewards, gamma)
        G = self.get_g_value(rewards, gamma)
        G_mean = np.mean(G)
        G_std = np.std(G)
        G = (G - G_mean)/G_std
        G = G * self.down_scale_factor
        # np.reshape(G,[total_step_num, 1])
        # G = self.get_g_value(rewards, gamma)
        # print(np.shape(G))
        # print(total_step_num)
        # loss = self.get_loss_value(G, log_value)
        # print(loss)
        # train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        # states = np.reshape(states,[total_step_num,self.observation_space_size])
        # actions = np.reshape(actions, [total_step_num, 1])
        # rewards = np.reshape(rewards, [total_step_num, 1])
        # print(actions)
        loss_value, _ , log_value, output= self.sess.run([self.loss, self.train_op, self.log_value, self.model.output], 
            {self.model.input : np.vstack(states), 
            self.actions_input: np.array(actions),  
            self.total_steps: float(total_step_num), 
            self.G_value: G})
        return loss_value, cumulative_reward, output


    # def get_log_value(self, states, actions):
    #     # self.sess.run(self.model.output, {self.model.input : states})
    #     # total_steps = np.shape(actions)[0]
    #     # print(total_steps)
    #     # actions = np.reshape(actions,[total_steps,1])
    #     # idx = np.reshape(range(total_steps),[total_steps,1])
    #     # print(idx[1])
    #     # print(np.shape(actions))
    #     # index = np.concatenate((idx, actions),axis = 1)
    #     # print(np.shape(index))
    #     # specific_policy = tf.gather_nd(self.model.output, index)
    #     depth = 4
    #     a_matrix = tf.one_hot(actions, depth)
    #     log_value = tf.log(specific_policy)
    #     return log_value



    def get_g_value(self, rewards, gamma=1.0):
        steps =  np.shape(rewards)[0]
        G = []
        # idx = steps - 1
        # while idx >= 0:
        #     t = idx
        #     G_t = 0
        #     k = t
        #     while k <= steps - 1:
        #         G_t += gamma ** (k - t) * rewards[k]
        #         k += 1
        #     G.append(G_t)
        #     idx -= 1
        # G = np.flip(G)
        # G_mean = np.mean(G)
        # G_std = np.std(G)
        # print(G_mean)
        # print(G_std)
        # G = (G - G_mean)/G_std
        G = []
        # get the discounted reward
        for time in range(steps):
            Gt = 0
            for k in range(time, steps):
                Gt += gamma**(k-time) * (rewards[k])
            G.append(Gt)
        np.reshape(np.array(G),[steps, 1])
        return G

    # def get_loss_value(self, G, log_value):
    #     loss = tf.reduce_mean(np.sum(G * log_value))
    #     return loss


    def generate_episode(self, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []
        current_state = self.env.reset()
        done = False
        while True:    # print(np.shape(states))    # print(np.shape(states))
            if render == True:
                self.env.render()
            states.append(current_state)
            # policy = self.sess.run(self.model.output, {self.model.input : np.reshape(current_state, [1,self.observation_space_size])})
            policy = self.model.predict(np.reshape(current_state, [1,self.observation_space_size]))
            # print(policy)
            current_action = np.random.choice([0,1,2,3], 
                                  p=policy[0])  
            # current_action = int(current_action)
            current_state, reward, done, info = self.env.step(current_action) #update state
            rewards.append(reward)
            actions.append(current_action)
            if done:
                break
        # print(policy)
        self.env.close()
        return states, actions, rewards


    def test(self, gamma = 1.0):
        test_episode = 100
        cumulative_reward = []
        for episode in range(test_episode):
            states, actions, rewards = self.generate_episode()
            G = self.get_g_value(rewards, gamma)
            cumulative_reward.append(G[0])
        cumulative_reward_mean = np.mean(cumulative_reward)
        cumulative_reward_std = np.std(cumulative_reward)
        return cumulative_reward_mean, cumulative_reward_std


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
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = 0.0003
    render = args.render
    test_episode = 1000
    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    # print(tf.shape(model.output))
    # print(tf.shape(model.input))
    # print(env.action_space)
    # print(env.observation_space)
    initial_state = env.reset()
    sess = tf.Session()
    # policy = sess.run(model.output, {model.input : np.reshape(initial_state, [1,8])})
    # print(policy)
    # print(np.random.choice(range(3), 1, list(policy[0])))
    reinforce = Reinforce(model, lr, env, sess, render)
    sess.run(tf.global_variables_initializer())
    # states, actions, rewards = reinforce.generate_episode(sess, render = True)
    # print(np.shape(states))
    # print(np.shape(rewards)[0])
    # G = reinforce.get_g_value(rewards)
    # print(G)
    # print(np.shape(G))
    # policy = sess.run(model.output, {model.input : states})
    # print(model.output)
    # a = tf.constant([[1],[2],[3]])
    # b = tf.constant([[1],[2],[3]])
    # c = a*b
    # print(sess.run(c))
    test_mean = []
    test_std = []
    for episode in range(num_episodes):
        loss, discounted_reward, output = reinforce.train(gamma = 0.99)
        # print("Episode: %d Loss: %f  Reward: %f" % (episode,loss,discounted_reward[0]))
        if episode % 5 == 0:
            print("Episode: %d Loss: %f  Reward: %f" % (episode,loss,discounted_reward[0]))
            # print(output)
            # print(discounted_reward)
        if episode % 200 == 0:
            states, actions, rewards = reinforce.generate_episode()
            # print(output)
            # print(actions)
        if episode % test_episode ==0:
            mean, std = reinforce.test(gamma = 0.99)
            print("Episode: %d mean: %f  std: %f" % (episode,mean,std))
            test_mean.append(mean)
            test_std.append(std)
    # loss, discounted_reward, output = reinforce.train(gamma = 1)
    # print(discounted_reward)
    fig, (axes) = plt.subplots(nrows=1)
    x = np.array(range(0, num_episodes, test_episode))
    y = np.array(test_mean)
    stds = np.array(test_std)
    axes.errorbar(x, y, yerr=stds, fmt='-o')
    plt.savefig('test.png')

if __name__ == '__main__':
    main(sys.argv)


#gamma = 0.99, lr = 0.0003, episode_test = 1000, test_episode = 100, without downscale factor, '-o'
