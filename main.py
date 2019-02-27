from collections import deque
import numpy as np
from DQN import Multi_DQN
from stuff_world import StuffWorld, TabularEnv
from itertools import count
from random import sample
import os
from gym import Env
from typing import List
from tabular_q_learner import TabularQLearner

state_buffer = deque(maxlen=100000)


def prepare_state_for_dqn(state):
    state = np.array(state).copy().astype(np.float32)
    state[-2:] /= 10.
    return state

def task_sort_key(task_name):
    collected_stuff = [int(c) for c in task_name]
    return (len(collected_stuff), sum(collected_stuff))

def load_q_table(env: TabularEnv, path : str):
    learner = TabularQLearner(env.produce_q_table(), env.action_space.n)
    learner.restore_q_values(path)
    return learner

def build_target_q_batch(tables: List[TabularQLearner], states, tasks, env: Env):
    target_qs = []
    for task, state in zip(states, tasks):
        target_q = [tables[task].Q[(state,a)] for a in range(env.action_space.n)]
        target_qs.append(target_q)
    return target_qs # [bs, num_actions]

def do_run():
    num_tasks = 20
    num_dqns = 10
    train_freq = 4
    min_buffer_size = 100
    q_func_dir = './q_funcs'
    task_names = sorted([f for f in os.listdir(q_func_dir) if f.isnumeric()], key=task_sort_key)
    task_names = task_names[:num_tasks]
    print(task_names)
    assert len(task_names) == num_tasks

    env = StuffWorld()
    tables = [load_q_table(env, os.path.join(q_func_dir, task_name)) for task_name in task_names]
    dqn = Multi_DQN(num_tasks, num_dqns, env, 'multi_dqn')
    env.reset()

    for i in count():
        a = np.random.randint(0, env.action_space.n)
        s, r, t, info = env.step(a)
        state_buffer.append(s)
        if len(state_buffer) >= min_buffer_size and i % train_freq == 0:
            states = sample(state_buffer, 32)
            tasks = np.random.randint(0, num_tasks, size=[32])
            target_qs = build_target_q_batch(tables, states, tasks, env)
            loss = dqn.train(states, tasks, target_qs)
            print(loss)

if __name__ == '__main__':
    do_run()