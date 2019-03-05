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
from multiprocessing import Pool, cpu_count

state_buffer = deque(maxlen=100000)


def prepare_state_for_dqn(state):
    state = np.array(state).copy().astype(np.float32)
    state[-2:] /= 10.
    return state

def task_sort_key(task_name):
    collected_stuff = [int(c) for c in task_name]
    return (len(collected_stuff), sum(collected_stuff))

def async_load_q_tables(env, path_list):
    pool = Pool(processes=cpu_count())
    q_tables = []
    for path in path_list:
        res = pool.apply_async(load_q_table, (env, path))
        q_tables.append(res)
    q_tables = [res.get() for res in q_tables]
    pool.close()
    pool.join()
    return q_tables


def load_q_table(env: TabularEnv, path : str, table_number: int):
    print(f'({table_number}) loading table for {path}...')
    learner = TabularQLearner(env.produce_q_table(), env.action_space.n)
    learner.restore_q_values(path)
    print(f'({table_number}) finished loading {path}!')
    return learner

def build_target_q_batch(tables: List[TabularQLearner], states, tasks, env: Env):
    target_qs = []
    for state, task in zip(states, tasks):
        target_q = [tables[task].Q[(state,a)] for a in range(env.action_space.n)]
        target_qs.append(target_q)
    return target_qs # [bs, num_actions]

def visualize_behavior(task_num, num_tasks, num_dqns):
    env = StuffWorld()
    #tabular_agent = TabularQLearner(env.produce_q_table(), env.action_space.n)
    #tabular_agent.restore_q_values('./q_funcs/1')

    env.set_goal_set(set(range(10)))
    dqn = Multi_DQN(num_tasks, num_dqns, env, 'multi_dqn')
    dqn.restore('./multi_dqn.ckpt')
    w = np.zeros([10], dtype=np.float32)
    w[task_num] = 1.0
    #w[4] = 1.0
    W = dqn.get_w()
    #for i in range(100):
    #    print(f'task {i}', W[:, i])
    #w = W[:,task_num]
    #print(w)

    s = env.reset()
    #print(env.visual())
    while True:
        #a = env.human_mapping[input('a:')]
        #print('tabular_qs', tabular_agent.get_Qs(s))

        s = prepare_state_for_dqn(s)
        a = dqn.get_action([s], w)[0]
        #a = tabular_agent.act(s)
        #print(a)
        s, _, _, _ = env.step(a)
        print(env.visual())
        input('------')

def sample_goal_set():
    return set([x for x in range(2) if np.random.uniform() < 0.5])

def sample_goal_num(num_tasks):
    return np.random.randint(0, num_tasks)





def do_run():

    num_tasks = 100
    num_dqns = 10
    train_freq = 4
    min_buffer_size = 1000
    save_interval = 100000
    dqn_path = './multi_dqn.ckpt'
    q_func_dir = './q_funcs'
    task_names = sorted([f for f in os.listdir(q_func_dir) if f.isnumeric()], key=task_sort_key)
    task_names = task_names[:num_tasks]
    task_sets = [set([int(x) for x in name]) for name in task_names]
    #print(task_sets)
    #input('...')
    #print(task_names)
    assert len(task_names) == num_tasks

    env = StuffWorld()
    current_goal_num = sample_goal_num(num_tasks)
    env.set_goal_set(task_sets[current_goal_num])
    paths = [os.path.join(q_func_dir, task_name) for task_name in task_names]
    #tables = async_load_q_tables(env, paths)
    tables = [load_q_table(env, path, table_number) for table_number, path in enumerate(paths)]
    dqn = Multi_DQN(num_tasks, num_dqns, env, 'multi_dqn', use_silencer=True)
    s = env.reset()

    for i in count():
        if np.random.uniform() < 0.8:
            #print(tables[0].get_Qs(s))
            a = np.argmax(tables[current_goal_num].get_Qs(s))
            #print(a)
        else:
            a = np.random.randint(0, env.action_space.n)
        s, r, t, info = env.step(a)
        if t:
            current_goal_num = sample_goal_num(num_tasks)
            env.set_goal_set(task_sets[current_goal_num])
            s = env.reset()
        state_buffer.append((s, current_goal_num))
        if len(state_buffer) >= min_buffer_size and i % train_freq == 0:
            states_and_tasks = sample(state_buffer, 32)
            states, tasks = zip(*states_and_tasks)
            #print(tasks)
            #tasks = np.random.randint(0, num_tasks, size=[32])
            target_qs = build_target_q_batch(tables, states, tasks, env)
            #print(target_qs)
            preped_states = [prepare_state_for_dqn(state) for state in states]
            loss = dqn.train(preped_states, tasks, target_qs)
            print(i, loss)

        if i % save_interval == 0:
            dqn.save(dqn_path)

if __name__ == '__main__':
    do_run()
    #w = np.zeros([10])
    #w[1] = -1.0
    #q_func_dir = './q_funcs'
    #task_names = sorted([f for f in os.listdir(q_func_dir) if f.isnumeric()], key=task_sort_key)
    #print(task_names[:100])
    #visualize_behavior(9, 100, 10)