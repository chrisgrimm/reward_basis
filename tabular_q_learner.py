from itertools import count
import dill as pickle
from stuff_world import StuffWorld
from itertools import count, combinations
from collections import deque
from random import sample
import numpy as np
import os
from multiprocessing import Pool, cpu_count

class TabularQLearner():

    def __init__(self, env_q_table, env_num_actions):
        self.Q = env_q_table
        self.num_actions = env_num_actions
        self.gamma = 0.99
        self.alpha = 0.1

    def store_q_values(self, filepath):
        sorted_qs = []
        with open(filepath, 'wb') as f:
            for key in sorted(self.Q.keys()):
                sorted_qs.append(self.Q[key])
            pickle.dump(sorted_qs, f)

    def restore_q_values(self, filepath):
        with open(filepath, 'rb') as f:
            sorted_qs = pickle.load(f)
            for key, q in zip(sorted(self.Q.keys()), sorted_qs):
                self.Q[key] = q

    def train(self, s, a, r, sp, t):
        max_q = max([self.Q[(sp, a_prime)] for a_prime in range(self.num_actions)])
        updated = r + ((0 if t else 1) * self.gamma * max_q)
        td_error = np.abs(self.Q[(s,a)] - updated)
        self.Q[(s, a)] = self.Q[(s, a)] + self.alpha*(updated - self.Q[(s, a)])
        return td_error
        #q = self.Q[(s,a)]
        #if q != 0:
        #    print(q)


    def act(self, s):
        return max([(i, self.Q[(s, i)]) for i in range(self.num_actions)],key=lambda x: x[1])[0]


def visualize_episode(env, q_learner):
    s = env.reset()
    for i in count():
        #if np.random.uniform() < 0.1:
        #    a = np.random.randint(0,4)
        #else:
        a = q_learner.act(s)
        s, r, t, info = env.step(a, debug=True)
        print(env.visual())
        print(f'{i}------r={r},t={t}')
        if i > 100 or t:
            break
    return i


def run_stuff_world(q_table_path, num_train_steps, goal_set):
    goal_name = ''.join([str(x) for x in sorted(list(goal_set))])
    print(f'Spinning up {goal_name}...')
    env = StuffWorld()
    env.set_goal_set(goal_set.copy())
    q_table = env.produce_q_table()
    q_learner = TabularQLearner(q_table, env.action_space.n)
    buffer = deque(maxlen=10000)
    s = env.reset()
    for _ in range(num_train_steps):
        # take a random action 80% of the time.
        if np.random.uniform(0, 1) < 0.8:
            a = np.random.randint(0,4)
        else:
            a = q_learner.act(s)
        sp, r, t, info = env.step(a)
        buffer.append((s,a,r,sp,t))
        _s,_a,_r,_sp, _t = sample(buffer, 1)[0]
        td = q_learner.train(_s, _a, _r, _sp, _t)
        if t:
            s = env.reset()
        else:
            s = sp
    q_learner.store_q_values(os.path.join(q_table_path, goal_name))
    print(f'Finished {goal_name}...')




if __name__ == '__main__':
    q_function_dir = './q_funcs'
    num_training_steps = 10_000_000
    pool = Pool(processes=cpu_count())
    for r in range(1, 10+1):
        for combination in combinations(range(10), r):
            goal_set = set(combination)
            pool.apply_async(run_stuff_world, (q_function_dir, num_training_steps, goal_set))
    pool.close()
    pool.join()




