import cv2
import gym
import numpy as np
from gym import spaces

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,1))

    def _observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(img):
        img = img[:,:,0] * 0.299 + img[:,:,1] * 0.587 +img[:,:,2] * 0.114
        x_t = cv2.resize(img, (84,84), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(x_t, (84, 84, 1))
        x_t = np.nan_to_num(x_t)
        return x_t.astype(np.uint8)


# class MarioActionSpaceWrapper(gym.ActionWrapper):
#     mapping = {
#         0: np.array([[0, 0, 0, 0, 0, 0]] * 4),  # NO
#         1: np.array([[1, 0, 0, 0, 0, 0]] * 4),  # Up
#         2: np.array([[0, 1, 0, 0, 0, 0]] * 4),  # Down
#         3: np.array([[0, 0, 1, 0, 0, 0]] * 4),  # Left
#         4: np.array([[0, 0, 1, 0, 1, 0]] * 4),  # Left + A
#         5: np.array([[0, 0, 1, 0, 0, 1]] * 4),  # Left + B
#         6: np.array([[0, 0, 1, 0, 1, 1]] * 4),  # Left + A + B
#         7: np.array([[0, 0, 0, 1, 0, 0]] * 4),  # Right
#         8: np.array([[0, 0, 0, 1, 1, 0]] * 4),  # Right + A
#         9: np.array([[0, 0, 0, 1, 0, 1]] * 4),  # Right + B
#         10: np.array([[0, 0, 0, 1, 1, 1]] * 4),  # Right + A + B
#         11: np.array([[0, 0, 0, 0, 1, 0]] * 4),  # A
#         12: np.array([[0, 0, 0, 0, 0, 1]] * 4),  # B
#         13: np.array([[0, 0, 0, 0, 1, 1]] * 4),  # A + B
#     }
#     def __init__(self, env):
#         super(MarioActionSpaceWrapper, self).__init__(env)
#         self.action_space = spaces.Discrete(14)
#
#     def _action(self, action):
#         return self.mapping.get(action)[0]
#
#     def _reverse_action(self, action):
#         for k in self.mapping.keys():
#             if(self.mapping[k] == action):
#                 return self.mapping[k]
#
