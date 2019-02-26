import numpy as np
import itertools
from gym.spaces import Box, Discrete
class StuffWorld(object):

    def __init__(self):
        self.height = 10
        self.width = 10
        self.num_stuff = 10
        self.stuff_ordering = [(0,0), (5,0), (9,0),
                               (0,3), (5,3), (9,3),
                               (0,6), (5,6), (9,6),
                                      (5,9)]
        self.stuff_locations = {
            (0,0): (True, 0),
            (5,0): (True, 1),
            (9,0): (True, 2),
            (0,3): (True, 3),
            (5,3): (True, 4),
            (9,3): (True, 5),
            (0,6): (True, 6),
            (5,6): (True, 7),
            (9,6): (True, 8),
            (5,9): (True, 9)
        }


        self.agent_location = (5,5)
        self.action_mapping = {
            0: (-1,0),
            1: (1,0),
            2: (0,-1),
            3: (0,1)
        }

        self.goal_set = set([0])

        self.remaining = self.goal_set.copy()
        self.action_space = Discrete(4)

    def pair_to_idx(self, x, y):
        return y * self.height + x

    def idx_to_pair(self, i):
        y = i // self.height
        x = i % self.width
        return (x, y)

    def generate_obs(self, alt_is_present=None, alt_agent_pos=None):
        canvas = np.zeros([self.height * self.width + 2], dtype=np.uint8)
        agent_x, agent_y = self.agent_location
        if alt_agent_pos is not None:
            agent_x, agent_y = alt_agent_pos
        for stuff_pair, (is_present, stuff_num) in self.stuff_locations.items():
            if alt_is_present is not None:
                is_present = alt_is_present[stuff_num]
            if is_present:
                canvas[self.pair_to_idx(*stuff_pair)] = 1
        canvas[-2] = agent_x
        canvas[-1] = agent_y
        return tuple(canvas)

    def produce_q_table(self):
        all_obs = dict()
        for a in range(4):
            for is_present_list in itertools.product([True, False], repeat=10):
                #print(is_present_list)
                for x_pos, y_pos in itertools.product(range(self.width), range(self.height)):
                    obs = self.generate_obs(alt_is_present=is_present_list, alt_agent_pos=(x_pos, y_pos))
                    all_obs[(obs, a)] = 0.0
        return all_obs

    def visual(self):
        canvas = np.zeros([self.height, self.width], dtype=np.uint8)
        for (stuff_x, stuff_y), (is_present, stuff_num) in self.stuff_locations.items():
            if is_present:
                canvas[stuff_y, stuff_x] = 2
        agent_x, agent_y = self.agent_location
        canvas[agent_y, agent_x] = 1
        return canvas


    def step(self, action, debug=False):
        delta_x, delta_y = self.action_mapping[action]
        old_x, old_y = self.agent_location
        new_x, new_y = np.clip(old_x + delta_x, 0, self.width-1), np.clip(old_y + delta_y, 0, self.height-1)
        if (new_x, new_y) in self.stuff_locations:
            (is_present, stuff_num) = self.stuff_locations[(new_x, new_y)]
            if is_present:
                r = 1 if stuff_num in self.goal_set else -1
                if stuff_num in self.remaining:
                    self.remaining.remove(stuff_num)
                #print(f'Grabbed {(new_x, new_y)}')
                self.stuff_locations[(new_x, new_y)] = (False, stuff_num)
                #print(self.stuff_locations)
            else:
                r = -0.01
        else:
            r = -0.01
        self.agent_location = (new_x, new_y)
        if len(self.remaining) == 0:
            t = True
        else:
            t = False
        return self.generate_obs(), r, t, dict()

    def reset(self):
        for stuff, (is_present, stuff_num) in self.stuff_locations.items():
            self.stuff_locations[stuff] = (True, stuff_num)
        self.agent_location = (5,5)
        self.remaining = self.goal_set.copy()
        return self.generate_obs()


    def set_goal_set(self, new_goal_set : set):
        self.goal_set = new_goal_set.copy()
        self.reset()
