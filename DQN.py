import numpy as np
import os
import tensorflow as tf
from gym import Env

class Multi_DQN:
    def __init__(self, num_tasks: int, num_dqns: int, env: Env, name: str, reuse=None,
                 use_silencer=False):
        assert len(env.observation_space.shape) == 1
        obs_size = env.observation_space.shape[0]
        self.num_dqns = num_dqns
        self.action_size = action_size = env.action_space.n
        self.inp_s = tf.placeholder(tf.float32, [None, obs_size])
        self.inp_target_q = tf.placeholder(tf.float32, [None, action_size])
        self.inp_task_indicator = tf.placeholder(tf.int32, [None])
        self.inp_w = tf.placeholder(tf.float32, [None, num_dqns])

        inp_task_indicator_onehot = tf.one_hot(self.inp_task_indicator, num_tasks) #[bs, num_tasks]
        lmbda = 0.1
        #bs = tf.shape(self.inp_task_indicator)[0]

        with tf.variable_scope(name, reuse=reuse) as scope:
            all_Q = [self.build_network(f'qnet{i}') for i in range(num_dqns)] # [num_dqns, bs, num_actions]
            all_Q = tf.transpose(all_Q, [1,0,2]) # [bs, num_dqns, num_actions]
            if use_silencer:
                silencer = self.build_silencing_network('silencer') # [bs, num_dqns]
                all_Q = tf.reshape(silencer, [-1, num_dqns, 1]) * all_Q
            self.Q_tilde_w = tf.reduce_sum(tf.reshape(self.inp_w, [-1, num_dqns, 1]) * all_Q, [1]) # [bs, num_actions]

            self.w = w = tf.get_variable('w', shape=[num_dqns, num_tasks])

            pre_selection = tf.reshape(w, [1, num_dqns, num_tasks]) * tf.reshape(inp_task_indicator_onehot, [-1, 1, num_tasks]) # [bs, num_dqns, num_tasks]
            selected_w = tf.reduce_sum(pre_selection, axis=2) # [bs, num_dqns]
            selected_Q_tildes = tf.reduce_sum(tf.reshape(selected_w, [-1, num_dqns, 1]) * all_Q, axis=1) # [bs, num_actions]
            loss = tf.losses.huber_loss(self.inp_target_q, selected_Q_tildes)
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(selected_Q_tildes - self.inp_target_q), axis=1), axis=0)
            reg = tf.reduce_mean(tf.square(selected_w))
            self.loss = loss = loss + lmbda * reg
            vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.original_name_scope)
            print(vars)
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(loss, var_list=vars)
            vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.original_name_scope)

            self.saver = tf.train.Saver(var_list=vars)

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.variables_initializer(vars))

    def get_w(self):
        return self.sess.run(self.w)

    def save(self, path):
        print(path)
        self.saver.save(self.sess, path)

    def restore(self, path):
        self.saver.restore(self.sess, path)

    def get_action_for_task(self, states, task_num):
        W = self.sess.run(self.w)
        w = W[:, task_num]
        return self.get_action(states, w)

    def get_action(self, states, w):

        [qs] = self.sess.run([self.Q_tilde_w], feed_dict={
            self.inp_s: states,
            self.inp_w: np.tile(np.reshape(w, [1, -1]), [len(states), 1])
        })
        print('qs', qs)
        return np.argmax(qs, axis=1)

    def train(self, states, task_nums, target_qs):
        [_, loss] = self.sess.run([self.train_op, self.loss], feed_dict={
            self.inp_s: states,
            self.inp_task_indicator: task_nums,
            self.inp_target_q: target_qs})
        return loss



    def build_network(self, name: str, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            fc1 = tf.layers.dense(self.inp_s, 512, tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 512, tf.nn.relu, name='fc2')
            qs = tf.layers.dense(fc2, self.action_size, name='qs')
        return qs

    def build_silencing_network(self, name: str, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            fc1 = tf.layers.dense(self.inp_s, 512, tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 512, tf.nn.relu, name='fc2')
            silencer = tf.layers.dense(fc2, self.num_dqns, tf.nn.sigmoid, name='qs') # [bs, num_dqns]
        return silencer


