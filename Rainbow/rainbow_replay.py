import numpy as np
import random
import tensorflow as tf

class SumTree:
    write = 0
    count = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total_and_count(self):
        return self.tree[0], self.count

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.count < self.capacity:
            self.count += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataidx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataidx]

    def get_data(self, dataidx):
        return self.data[dataidx]

class PriorityExperienceReplay:
    '''
    Almost copy from
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    '''
    def __init__(self,
                args):
        self.tree = SumTree(args.memory_capacity)
        self._max_size = args.memory_capacity
        self.num_step = args.num_step
        self.discount = args.gamma
        self._window_size = args.window_size
        self.priority_weight = args.priority_weight
        self._WIDTH = args.input_shape[0]
        self._HEIGHT = args.input_shape[1]
        self.e = 0.01
        self.a = 0.6
        self.t = 0

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def append(self, old_state, action, reward, new_state, is_terminal):
        for o_s, a, r, n_s, i_t in zip(old_state, action, reward, new_state, is_terminal):
            # 0.5 is the maximum error
            p = self._getPriority(0.5)
            self.tree.add(p, data=(self.t, o_s, a, r, n_s, i_t)) 
            # self.t = 0 if i_t else self.t += 1


    def _get_transitions(self, dataidx):
        transition = [None] * (self._window_size + self.num_step)
        transition[self._window_size - 1] = self.tree.get_data(dataidx) # 3
        for i in range(self._window_size - 2, -1, -1): # 2, 1, 0
            if transition[i + 1][0] == 0:
                transition[i] = tf.zeros_like(transition[self._window_size - 1])
            else:
                transition[i] = self.tree.get_data(dataidx - self._window_size + i + 1)
        for i in range(self._window_size, self._window_size + self.num_step): # 4, 5, 6
            if transition[i - 1][4] == False:
                transition[i] = self.tree.get_data(dataidx - self._window_size + i + 1)
            else:
                transition[i] = tf.zeros_like(transition[self._window_size - 1])
        return transition

    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = random.uniform(i * segment, (i + 1) * segment)
            idx, prob, data = self.tree.get(sample)
            dataidx = idx - self._max_size + 1
            if (self.tree.count - idx) % self._max_size > self.num_step and (idx - self.tree.count) % self._max_size >= self._window_size and prob != 0:
                valid = True

        transition = self._get_transitions(dataidx)
        state = [trans[1] for trans in transition[:self._window_size]] #), tf.float32) # /255
        next_state = [trans[1] for trans in transition[self.num_step : self.num_step + self._window_size]] # /255
        action = [transition[self._window_size - 1][2]]
        reward = [sum(self.discount ** n * transition[self._window_size + n - 1][3] for n in range(self.num_step))] # 1dim
        is_terminal = [transition[self._window_size + self.num_step - 1][-1]]
        return prob, dataidx, idx, state, action, reward, next_state, is_terminal

    def sample(self, batch_size):
        p_total, count = self.tree.total_and_count()
        segment = p_total / batch_size
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]
        probs, dataidxs, idxs, states, actions, rewards, next_states, is_terminals = zip(*batch)
        
        states = np.reshape(states, (-1, self._WIDTH, self._HEIGHT, self._window_size))
        next_states = np.reshape(next_states, (-1, self._WIDTH, self._HEIGHT, self._window_size))
        # actions, returns, is_terminals =
        probs = probs / p_total
        weights = (count * probs) ** -self.priority_weight
        weights = weights / max(weights)
        return idxs, states, actions, rewards, next_states, is_terminals, weights

    def update(self, idx_list, error_list):
        for idx, error in zip(idx_list, error_list):
            p = self._getPriority(error)
            self.tree.update(idx, p)
