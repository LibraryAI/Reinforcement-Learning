import args..
from PIL import Image

class Agent():
	def __init__(self, args, env):
		self.action_size = env.actions_size
		self.batch_size = args.batch_size
		self.discount = args.gamma
		self.num_iteration = args.num_iteration
		self.update_freq = args.replay_frequency
		self._learning_rate = args.learning_rate
		self.learn_start = args.learn_start
		self._rmsp_decay = args.rmsp_decay
		self._rmsp_momentum = args.rmsp_momentum
		self._rmsp_epsilon = args.rmsp_epsilon
		self.num_step = args.num_step
		self.atoms = args.atoms #51
		self.V_max = args.V_max #20.0
		self.V_min = args.V_min #0.0
		self.Delta_z = (args.V_max - args.V_min)/(args.atoms - 1)
		self.support = tf.constant([args.V_min + i * self.Delta_z for i in range(args.atoms)], dtype = tf.float32)
		self.support_broadcasted = tf.tile(tf.reshape(self.support, [1, args.atoms]), tf.constant([self.action_size, 1])) # action_size 어디서 search할걸지 고려하기
		self._action_ph = tf.placeholder(tf.int32, [None, 2], name ='action_ph')
        self._reward_ph = tf.placeholder(tf.float32, name='reward_ph')
        self._is_terminal_ph = tf.placeholder(tf.float32, name='is_terminal_ph')
        #self._action_chosen_by_online_ph = tf.placeholder(tf.int32, [None, 2], name ='action_chosen_by_online_ph')
        self._loss_weight_ph = tf.placeholder(tf.float32, name='loss_weight_ph')
        self.priority_weight_increase = (1 - args.priority_weight) / (args.num_iteration - args.learn_start)

		# parameter 선언
		start_actor()

	def start_actor(self):
		# learner 로 부터 parameter를 받는 target, behavior model 생성
		self.behavior_n = rainbow_net(args)
		self.behavior_n.build_model(trainable = True, log = False, flag = "all")
		self.target_n = rainbow_net(args, action_size)
		self.target_n.build_model(trainable = False, log = False, flag = "all")

	def update_target(self):
		# target_model 의 파라미터를 behavior 모델의 파라미터로 교체		
		for name in self.target_n.w.keys():
			self.target_n.w[name].assign(self.behavior_n.w[name])
	
	def reset_noise(self):
		# behavior model의 noise 재생성
		self.behavior_n.reset_noise()

	def act(self, sess, state):
		#predict() 를통해서 얻은 |A| * |Atom| 크기의 행렬로 부터 Atom은 다 sum, |A| 중에 e-greedy로
		#한가지 action을 output
		state = state.astype(np.float32) / 255.0
		feed_dict = {self.behavior_n.input_frames : state}

		q_network = tf.multiply(self.behavior_n.output, self.support_broadcasted)
		q_network = tf.reduce_sum(q_network, axis = 2, name = 'q_values')
		action = tf.argmax(q_network, axis = 1)
		action = mapping[action]
		action = sess.run(action, feed_dict = feed_dict)
		return action

	def act_e_greedy(self, sess, state, epsilon = 0.001):
		return mapping[np.random.randint(0, self.action_size)] if np.random.rand() < epsilon else self.act(state)

	def evaluate_q(self, state):
		# distributional calculation을 통해 Q value 도출
		state = state.astype(np.float32) / 255.0
		feed_dict = {self.behavior_n.input_frames : state}

		q_network = tf.multiply(self.behavior_n.output, self.support_broadcasted)
		q_network = tf.reduce_sum(q_network, axis = 2, name = 'q_values')
		q_value = tf.reduce_max(q_network, axis = 1)
		return q_value

	def train(self, sess, env, fit_iteration):
		# 1. ReplayMemory로부터 sample batch를 받아오기
		#	memory로부터 multi-step 의 정도에 따른 s,a,r,s,a trajectory sample batch를 받아오기 
		#	mem class 가 memory 고, prioritiezed queue 로 이미 구현되어 있을때, smaple은 그에 바탕해서 뽑아온다고 가정
		#	{'state': , 'action': , 'reward': }
		is_terminal = False
        step_count = 0
        state = env.reset()
        score = 0
        distance = 0
        #prev_output = -1
        #repeat =0	
		state = env.reset()
		
		for t in range(0, fit_iteration):
			action = self.act(sess, state)
			for n in range(len(action)):
				next_state, reward, is_terminal, info = env.step(action[n])
				if is_terminal == True:
					break
			prev_distance = distance
            distance = info['distance']
            got_distance = distance-prev_distance

            past_score = score
            score = info['score']
            got_score = score-past_score

            time = info['time']
                   
            reward = got_score/50 + got_distance/30
                    
            if reward>0:
            	print("reward:", reward)

			if is_terminal: # Penalty
                        #time = info['time']
            	reward += -1.0

            if distance>=3000:
            	reward = 1
            #reward += distance / 1000
            print("last reward:", reward)

			mem.append(state, action, reward, next_state, is_terminal)
			
			if t % self.update_freq == 0:
				self.reset_noise()
			if t >= self.learn_start and t % self.update_freq == 0:
				idxs, states, actions, rewards, next_states, is_terminals, weights = mem.sample(self.batch_size)
				states, next_states = states.astype(np.float32) / 255.0. next_states.astype(np.float32) / 255.0
				actions, rewards, is_terminals = list(enumerate(actions)), np.array(rewards).astype(np.float32), np.array(is_terminals).astype(np.float32)
				feed_dict = {self.behavior_n.input_frames : states
							 self.target_n.input_frames : new_states,
							 self._action_ph : actions,
							 self._reward_ph : rewards,
							 self._is_terminal_ph : is_terminals,
							 self._loss_weight_ph : weights}
							 #self._action_chosen_by_online_ph : }

				target = tf.tile(tf.reshape(_reward_ph, [-1, 1]), tf.constant([1, self.atoms])) \
						+ (self.discount ** self.num_step) * tf.multiply(tf.reshape(self.support, [1, self.atoms]),
							(1.0 - tf.tile(tf.reshape(self._is_terminal_ph, [-1, 1]), tf.constant([1, self.atoms]))))
				target = tf.clip_by_value(target, self.V_min, self.V_max)
				b = (target - self.V_min) / self.Delta_z
				u = tf.ceil(b) ; l = tf.floor(b)
				u_id = tf.cast(u, tf.int32) ; l_id = tf.cast(l, tf.int32)
				u_minus_b = u - b ; b_minus_l = b - l

				pns = sess.run(self.behavior_n.output, feed_dict = {self.behavior_n.input_frames : new_states}
				q_network = tf.multiply(pns, self.support_broadcasted) #q-network
				q_network = tf.reduce_sum(q_network, axis = 2, name = 'q_values')
				action_indices = tf.argmax(q_network, axis = 1)
				self.target_n.reset_noise()
				pns = self.target_n.output
				pns_a = tf.gather_nd(pns, action_indices)

				ps = self.behavior_n.output
				ps_a = tf.gather_nd(ps, self._action_ph)

		        index_help = tf.tile(tf.reshape(tf.range(self.batch_size),[-1, 1]), tf.constant([1, self.atoms])) 
		        index_help = tf.expand_dims(index_help, -1)
		        u_id = tf.concat([index_help, tf.expand_dims(u_id, -1)], axis=2)
		        l_id = tf.concat([index_help, tf.expand_dims(l_id, -1)], axis=2)
		        error = pns_a * u_minus_b * tf.log(tf.gather_nd(ps_a, l_id)) + pns_a * b_minus_l * tf.log(tf.gather_nd(ps_a, u_id))
		        error = tf.reduce_sum(error, axis=1)
	       		loss = tf.negative(error * self._loss_weight_ph)
    
		        train_op = tf.train.RMSPropOmtimizer(self._learning_rate, decay = self._rmsp_decay, momentum=self._rmsp_momentum, epsilon=self._rmsp_epsilon).minimize(loss)
		        error_op = tf.abs(error, name='abs_error')
		        error, _ = sess.run([error_op, train_op], feed_dict = feed_dict)
		        mem.update(idx, error)
		        mem.priority_weight = min(mem.priority_weight + self.priority_weight_increase, 1)
		    
		    if is_terminal == True:
				is_terminal = False
		        step_count = 0
		        state = env.reset()
		        score = 0
		        distance = 0

