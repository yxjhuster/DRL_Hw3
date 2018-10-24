import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce_policy_gradient import Reinforce

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, sess, model, lr, critic_lr, num_episodes, render, env, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.state_dimension = 8
        self.action_dimention = 4
        self.model = model
        # self.critic_model = critic_model # need to define the critic model here
        self.n = n

        self.gamma = 0.99
        self.lr = lr
        self.critic_lr = critic_lr
        self.num_episodes = num_episodes
        self.env = env
        self.scale_factor = 1e-2
        self.render = render
        self.sess = sess
        # self.Reinforce = Reinforce(self.sess, model=self.model, lr=self.lr, num_episodes=self.num_episodes, render=self.render, env=self.env)

        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)

        self.actor_log_layer = tf.log(self.model.output)
        self.action_input = tf.placeholder(tf.int64)
        # self.actor_V_value = tf.placeholder(tf.float32)
        # self.actor_R_value = tf.placeholder(tf.float32)
        self.actor_advantage_value = tf.placeholder(tf.float32)
        self.actor_step_num = tf.placeholder(tf.float32)
        self.depth = 4
        self.actor_one_hot = tf.one_hot(self.action_input, self.depth)

        # self.critic_rewards = tf.placeholder(tf.float32)
        # self.critic_step_num = tf.placeholder

        self.critic_input = tf.placeholder(tf.float32, shape = (None, self.state_dimension))
        self.critic_output_labels = tf.placeholder(tf.float32) # dimension should be 1

        self.critic_hidden_layer1 = tf.layers.dense(self.critic_input, 128, tf.nn.relu)
        self.critic_hidden_layer2 = tf.layers.dense(self.critic_hidden_layer1, 128, tf.nn.relu)
        self.critic_output_layer = tf.layers.dense(self.critic_hidden_layer2, 1)

        # self.actor_loss = -tf.divide(tf.reduce_sum((self.actor_R_value - self.actor_V_value) * \
        #                         (self.actor_log_layer * self.actor_one_hot)), self.actor_step_num)
        self.actor_loss = -tf.divide(tf.reduce_sum(self.actor_advantage_value * \
                                    (self.actor_log_layer * self.actor_one_hot)), self.actor_step_num)
        self.critic_loss = tf.losses.mean_squared_error(self.critic_output_labels, self.critic_output_layer)

        self.actor_train_op = self.actor_optimizer.minimize(self.actor_loss)
        self.critic_train_op = self.critic_optimizer.minimize(self.critic_loss)

        # build initializer
        initializer = tf.global_variables_initializer()
        self.sess.run(initializer)

        # plot parameters
        self.num_test_episode = 100
        self.num_ploting_episode = 1000
        self.mean_list = []
        self.std_list = []

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  

    def train(self):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        for idx_episode in range(self.num_episodes):
            states, actions, rewards = self.generate_episode()

            total_step_num = len(states)
            R = []
            # get the R value
            for time in range(total_step_num):
                Rt = 0
                if (time + self.n) >= total_step_num:
                    V_end = 0
                else:
                    V_end = self.sess.run(self.critic_output_layer, feed_dict = {
                        self.critic_input: self.adjust_obs_format(states[time + self.n])
                        })
                Rt += V_end
                for idx in range(self.n):
                    if (time + idx) >= total_step_num:
                        Rt += 0 * (self.gamma ** idx)
                    else:
                        Rt += rewards[time + idx] * (self.gamma ** idx)
                R.append(Rt)

            R = np.reshape(np.array(R), [total_step_num,1])
            # might have dimension problem here
            value_predict = self.sess.run(self.critic_output_layer, feed_dict = {
                        self.critic_input: np.vstack(states)
                        })
            # print(value_predict)
            # print(value_predict.shape)
            # print(R.shape)
            # print(R)
            advantage_value = R - value_predict / self.scale_factor
            advantage_mean, advantage_std = np.mean(advantage_value), np.std(advantage_value)
            advantage_reduced = (advantage_value-advantage_mean) / advantage_std
            advantage_reduced = np.array(advantage_reduced).reshape([total_step_num, 1])

            actor_loss, _ = self.sess.run([self.actor_loss, self.actor_train_op], feed_dict ={
                self.model.input: np.vstack(states),
                self.action_input: np.array(actions),
                # self.actor_R_value: R * self.scale_factor,
                # self.actor_V_value: value_predict * self.scale_factor,
                self.actor_advantage_value: advantage_reduced,
                self.actor_step_num: float(total_step_num)               
                })

            critic_loss, _ = self.sess.run([self.critic_loss, self.critic_train_op], feed_dict = {
                self.critic_input: np.vstack(states),
                self.critic_output_labels: R * self.scale_factor
                })

            reward_episode = np.sum(rewards)

            print('Episode %d, Reward %.1f, actor_loss %.7f, critic_loss %.7f' % 
                                        (idx_episode, reward_episode, actor_loss,critic_loss))

            if idx_episode % self.num_ploting_episode == 0:
                self.test()

        self.plot()
        return


    def adjust_obs_format(self, observation):
        return observation.reshape(1, self.env.observation_space.shape[0])


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

            cumulative_reward.append(G[0])
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
        plt.savefig('test_a2c.png')


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

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
    critic_lr = 3*0.0001
    n = 20
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())
    f.close()

    # TODO: Train the model using A2C and plot the learning curves.
    a2c_agent = A2C(sess, model, lr, critic_lr, num_episodes, render, env, n)
    a2c_agent.train()


if __name__ == '__main__':
    main(sys.argv)

# gamma = 0.99
#actor: lr = 0.0003
#critic: lr = 0.0003, two hidden layers with 128 nodes, relu,
#episode_test = 1000, test_episode = 100
