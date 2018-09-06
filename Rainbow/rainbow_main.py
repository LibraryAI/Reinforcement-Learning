from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import numpy as np
import tensorflow as tf
import random
import gym
#from gym import wrappers

from ppaquette_gym_super_mario


env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
env = wrappers.Monitor(env, 'gym-results', force=True)
input_size = np.array([env.observation_space.shape[0], env.observation_space.shape[1], 15])

def bot_play(model, env=env):
 	  # See our trained network in action
   	state = env.reset()
  	reward_sum = 0
   	while True:
       	if state is None or state.size == 1:
           	output = randint(0, output_size - 1)
           	action = mapping[output]
           	print("random action:", output)
       	else:
       		output = model.act(state)
			action = mapping[output]           	
           	print("predicted action:", output)
       	for n in range(len(action)):
           	state, reward, done, info = env.step(action[n])
           	if done == True:
              	break
       	reward_sum += reward
       	if done:
           	print("Total score: {}".format(reward_sum))
           	break

def main():
	parser = argparse.ArgumentParser()
#parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')	
#parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
#parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
#parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
#parser.add_argument('--render', action='store_true', help='Display screen (testing only)')

	NUM_FRAME_PER_ACTION = 4
	#RMSP_EPSILON = 0.01
	#RMSP_DECAY = 0.95
	#RMSP_MOMENTUM =0.95
	NUM_FIXED_SAMPLES = 10000
	NUM_BURN_IN = 50000
	LINEAR_DECAY_LENGTH = 4000000

    parser = argparse.ArgumentParser(description='Rainbow on Mario')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--seed', default=10703, type=int, help='Random seed')
    parser.add_argument('--input_shape', default=(84,84), help='Input shape')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', default=0.1, help='Exploration probability in epsilon-greedy')
    parser.add_argument('--learning_rate', default=0.0000625, help='Training learning rate.')
    parser.add_argument('--window_size', default=4, type = int, help='Number of frames to feed to the Q-network')
    parser.add_argument('--batch_size', default=32, type = int, help='Batch size of the training part')
    #parser.add_argument('--num_process', default=3, type = int, help='Number of parallel environment')
    parser.add_argument('--num_iteration', default=20000000, type = int, help='number of iterations to train')
    parser.add_argument('--eval_every', default=0.001, type = float, help='What fraction of num_iteration to run between evaluations.')
    parser.add_argument('--num_step', default=1, type = int, help='Num Step for multi-step DQN, 3 is recommended')
    parser.add_argument('--atoms', type=int, default=51, help='Discretised size of value distribution')
    parser.add_argument('--hidden-size', type=int, default=512, help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.1, help='Initial standard deviation of noisy linear layers')
	parser.add_argument('--atoms', type=int, default=51, help='Discretised size of value distribution')
	parser.add_argument('--V-min', type=float, default=-10, help='Minimum of value distribution support')
	parser.add_argument('--V-max', type=float, default=10, help='Maximum of value distribution support')
	parser.add_argument('--history-length', type=int, default=4, help='Number of consecutive states processed')
	parser.add_argument('--max-episode-length', type=int, default=int(108e3), help='Max episode length (0 to disable)')
	parser.add_argument('--memory-capacity', type=int, default=int(1e6), help='Experience replay memory capacity')
	parser.add_argument('--replay-frequency', type=int, default=4, help='Frequency of sampling from memory')
	parser.add_argument('--target-update', type=int, default=10000, help='Number of steps after which to update target network')
	parser.add_argument('--evaluation-episodes', type=int, default=20, help='Number of evaluation episodes to average over')
	parser.add_argument('--learn-start', type=int, default=int(80e3), metavar='STEPS', help='Number of steps before starting training')
	parser.add_argument('--log-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between logging status')
	parser.add_argument('--rmsp-decay', type=float, default=0.95, metavar='')
	parser.add_argument('--rmsp-momentum', type=float, default=0.95, metavar='')
	parser.add_argument('--rmsp-epsilon', type=float, default=0.01, metavar='')

	parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
	parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
	parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
	args = parser.parse_args()

    action_idx = {
        0: 273,
        1: 274,
        2: 276,
        3: 275,  # right
		4: 97,
		5: 115
    }
	mapping = {
		0: np.array([[0, 0, 0, 0, 0, 0]]*4),  # NO
        1: np.array([[1, 0, 0, 0, 0, 0]]*4),  # Up
        2: np.array([[0, 1, 0, 0, 0, 0]]*4),  # Down
        3: np.array([[0, 0, 1, 0, 0, 0]]*4),  # Left
        4: np.array([[0, 0, 1, 0, 1, 0]]*4),  # Left + A
        5: np.array([[0, 0, 1, 0, 0, 1]]*4),  # Left + B
        6: np.array([[0, 0, 1, 0, 1, 1]]*4),  # Left + A + B
        7: np.array([[0, 0, 0, 1, 0, 0]]*4),  # Right
        8: np.array([[0, 0, 0, 1, 1, 0]]*4),  # Right + A
        9: np.array([[0, 0, 0, 1, 0, 1]]*4),  # Right + B
        10: np.array([[0, 0, 0, 1, 1, 1]]*4),  # Right + A + B
        11: np.array([[0, 0, 0, 0, 1, 0]]*4),  # A
        12: np.array([[0, 0, 0, 0, 0, 1]]*4),  # B
        13: np.array([[0, 0, 0, 0, 1, 1]]*4),  # A + B
	}
	iter_count = 1
	action_size = 13
    action_n = len(mapping.keys())
    color_chanel = 1
    state_n = resize_x * resize_y * color_chanel
	config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
	env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
	env = wrappers.Monitor(self.env, 'gym-results', force=True)
    mem = PriorityExperienceReplay()
    input_size = np.array([env.observation_space.shape[0], env.observation_space.shape[1], 15])    
    agent = Agent()
    retrain = False
    checkpoint_dir = './choong_checkpoint/'

    sess = tf.Session(config=config)
    with sess.as_default():
    	sess.run(tf.global_variables_initializer())
    	#tf.global_variables_initializer().run()
        sess.run(rainbow.update_target())
        if retrain = True:
        	saver = tf.train.Saver()
			saver.restore(sess, checkpoint_dir + 'model_notime.ckpt')
        	print("restored!")
    	


		fit_iteration = int(args.num_iteration * args.eval_every)

        for i in range(0, args.num_iteration, fit_iteration):
            # Evaluate:
            reward_mean, reward_var = agent.evaluate(sess, batch_environment, NUM_EVALUATE_EPSIODE)
            mean_max_Q = agent.get_mean_max_Q(sess, fixed_samples)
            print("%d, %f, %f, %f"%(i, mean_max_Q, reward_mean, reward_var))
            # Train:
            agent.train(sess, env, fit_iteration)
            if iter_count % 10 == 0:
            	saver = tf.train.Saver()
            	saver.save(sess, checkpoint_dir + 'model_notime.ckpt')
            	print("model saved")
            iter_count += 1

        for i in range(200):
        	bot_play(agent, env=env)

        env.close()
