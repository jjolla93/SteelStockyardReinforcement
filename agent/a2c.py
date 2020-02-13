"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
The cart pole example. Policy is oscillated.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow as tf
import os

from agent.helper import *
import environment.steelstockyard as ssy
import environment.plate as plate

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


class Actor(object):
    def __init__(self, sess, n_features, n_actions, feature_size, lr=0.001):
        self.sess = sess
        self.feature_size = feature_size

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.s_reshaped = tf.reshape(self.s, shape=[-1, self.feature_size[0], self.feature_size[1], 1])
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            conv1 = tf.layers.conv2d(
                inputs=self.s_reshaped,
                filters=16,
                kernel_size=4,
                strides=2,
                padding='VALID',
                activation=tf.nn.relu
            )

            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=32,
                kernel_size=2,
                strides=1,
                padding='VALID',
                activation=tf.nn.relu
            )
            conv2_flat = tf.layers.flatten(conv2)

            l1 = tf.layers.dense(
                inputs=conv2_flat,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l2,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., 0.3),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int

    def save_model(self):
        if not os.path.exists('../models/a2c'):
            os.makedirs('../models/a2c')
        if not os.path.exists('../models/a2c/actor'):
            os.makedirs('../models/a2c/actor')
        if not os.path.exists('../models/a2c/actor/%d-%d' % (self.feature_size[0], self.feature_size[1])):
            os.makedirs('../models/a2c/actor/%d-%d' % (self.feature_size[0], self.feature_size[1]))
        saver = tf.train.Saver()
        saver.save(self.sess, '../models/a2c/actor/%d-%d' % (self.feature_size[0], self.feature_size[1]))


class Critic(object):
    def __init__(self, sess, n_features, n_actions, feature_size, lr=0.01):
        self.sess = sess
        self.feature_size = feature_size

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.s_reshaped = tf.reshape(self.s, shape=[-1, feature_size[0], feature_size[1], 1])
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            conv1 = tf.layers.conv2d(
                inputs=self.s_reshaped,
                filters=16,
                kernel_size=4,
                strides=2,
                padding='VALID',
                activation=tf.nn.relu
            )

            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=32,
                kernel_size=2,
                strides=1,
                padding='VALID',
                activation=tf.nn.relu
            )
            conv2_flat = tf.layers.flatten(conv2)

            l1 = tf.layers.dense(
                inputs=conv2_flat,
                units=15,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., 0.3),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=15,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., 0.3),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            self.v = tf.layers.dense(
                inputs=l2,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

            with tf.variable_scope('squared_TD_error'):
                self.td_error = self.r + GAMMA * self.v_ - self.v
                self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_: v_, self.r: r})
        return td_error

    def save_model(self):
        if not os.path.exists('../models/a2c'):
            os.makedirs('../models/a2c')
        if not os.path.exists('../models/a2c/critic'):
            os.makedirs('../models/a2c/critic')
        if not os.path.exists('../models/a2c/critic/%d-%d' % (self.feature_size[0], self.feature_size[1])):
            os.makedirs('../models/a2c/critic/%d-%d' % (self.feature_size[0], self.feature_size[1]))
        saver = tf.train.Saver()
        saver.save(self.sess, '../models/a2c/critic/%d-%d' % (self.feature_size[0], self.feature_size[1]))


def plot_reward(rewards):
    if not os.path.exists('../summary/a2c/{0}_{1}/'.format(s_shape[0], s_shape[1])):
        os.makedirs('../summary/a2c/{0}_{1}/'.format(s_shape[0], s_shape[1]))
    import csv
    f = open('../summary/a2c/{0}_{1}/rewards_{2}_{3}.csv'.
             format(s_shape[0], s_shape[1], env.action_space, env.max_stack), 'w', encoding='utf-8')
    wr = csv.writer(f)
    for i in range(1, len(rewards)+1):
        wr.writerow([i, rewards[i-1]])
    f.close()
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(rewards)), rewards)
    plt.ylabel('average reward')
    plt.xlabel('training episode')
    plt.show()


def run(episodes=1000):
    step = 0
    rewards = []
    avg_rewards = []
    for episode in range(1, episodes+1):
        s = env.reset(episode)
        rs = []
        episode_frames = []
        while True:
            episode_frames.append(s)

            a = actor.choose_action(s)
            s_, r, done = env.step(a)
            rs.append(r)

            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

            s = s_

            if done:
                rewards.append(sum(rs))
                avg_rewards.append(sum(rewards) / len(rewards))
                break
            step += 1

        if episode % 1000 == 0:
            print('episode: {0} finished'.format(episode))
            if not os.path.exists('../frames/a2c'):
                os.makedirs('../frames/a2c')
            if not os.path.exists('../frames/a2c/%d-%d' % s_shape):
                os.makedirs('../frames/a2c/%d-%d' % s_shape)
            save_gif(episode_frames, s_shape, episode, 'a2c')
    actor.save_model()
    critic.save_model()
    plot_reward(avg_rewards)
    # end of game
    print('game over')


if __name__ == "__main__":

    # inbounds = plate.import_plates_schedule('../environment/data/plate_example1.csv')
    # inbounds = plate.import_plates_schedule_rev('../environment/data/SampleData.csv')
    inbounds = plate.import_plates_schedule_by_week('../environment/data/SampleData.csv')

    max_stack = 11
    num_pile = 6
    observe_inbounds = True
    if observe_inbounds:
        s_shape = (max_stack, num_pile + 1)
    else:
        s_shape = (max_stack, num_pile)
    s_size = s_shape[0] * s_shape[1]
    env = ssy.Locating(max_stack=max_stack, num_pile=num_pile, inbound_plates=inbounds,
                        observe_inbounds=observe_inbounds, display_env=False)

    N_F = s_size
    N_A = env.action_space

    # Superparameters
    #OUTPUT_GRAPH = False
    #RENDER = True  # rendering wastes time
    GAMMA = 0.99  # reward discount in TD error
    LR_A = 1e-5  # learning rate for actor
    LR_C = 1e-5  # learning rate for critic

    sess = tf.Session()

    actor = Actor(sess, N_F, N_A, s_shape, lr=LR_A)
    critic = Critic(sess, N_F, N_A, s_shape, lr=LR_C)

    sess.run(tf.global_variables_initializer())

    run(30000)