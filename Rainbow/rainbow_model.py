# Factorised Noisy linear layer with bias

	class rainbow_net():
		def __init__(self, args):
			# from args
			self.atoms = args.atoms # 51
			self.layer_design = args.layer_design
			self.window = args.window
			self.input_shape = args.input_shape
			self.action_size = 13
			self.history_length = args.history_length
			self.layer_design = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
			# self.conv_strides = 

        	

		def build_model(self, trainable, log = False, flag = "all" , model_name = "behavior"):

			self.w = {}
			
			# input_frames는 placeholder로 나중에 feed_dict로 실제값 받기
			with tf.variable_scope(model_name, reuse = tf.AUTO_REUSE):

				self.input_frames = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1], self.window], name = 'input_frames')
				
				# convolution layers 	
				self.conv1, self.w['conv1_w'], self.w['conv1_b'] = self.convolution_layer(self.input_frames, layer_num = 1, weight_shape = [4, 4, args.history_length, self.layer_design[0]], conv_strides = [1,4,4,1], trainable = trainable)
				self.conv2, self.w['conv2_w'], self.w['conv2_b'] = self.convolution_layer(self.conv1, layer_num = 2, weight_shape = [4, 4, self.layer_design[0], self.layer_design[1]], conv_strides = [1,2,2,1], trainable = trainable)
				self.conv3, self.w['conv3_w'], self.w['conv3_b'] = self.convolution_layer(self.conv2, layer_num = 3, weight_shape = [4, 4, self.layer_design[1], self.layer_design[2]], conv_strides = [1,2,2,1], trainable = trainable)
				self.conv4, self.w['conv4_w'], self.w['conv4_b'] = self.convolution_layer(self.conv3, layer_num = 4, weight_shape = [4, 4, self.layer_design[2], self.layer_design[3]], conv_strides = [1,2,2,1], trainable = trainable)
				self.conv5, self.w['conv5_w'], self.w['conv5_b'] = self.convolution_layer(self.conv4, layer_num = 5, weight_shape = [4, 4, self.layer_design[3], self.layer_design[4]], conv_strides = [1,2,2,1], trainable = trainable)
				self.conv6, self.w['conv6_w'], self.w['conv6_b'] = self.convolution_layer(self.conv5, layer_num = 6, weight_shape = [4, 4, self.layer_design[4], self.layer_design[5]], conv_strides = [1,2,2,1], trainable = trainable)
				self.conv7, self.w['conv7_w'], self.w['conv7_b'] = self.convolution_layer(self.conv6, layer_num = 7, weight_shape = [4, 4, self.layer_design[5], self.layer_design[6]], conv_strides = [1,2,2,1], trainable = trainable)
				self.conv8, self.w['conv8_w'], self.w['conv8_b'] = self.convolution_layer(self.conv7, layer_num = 8, weight_shape = [4, 4, self.layer_design[6], self.layer_design[7]], conv_strides = [1,2,2,1], trainable = trainable)
				self.conv9, self.w['conv9_w'], self.w['conv9_b'] = self.convolution_layer(self.conv8, layer_num = 9, weight_shape = [4, 4, self.layer_design[7], self.layer_design[8]], conv_strides = [1,2,2,1], trainable = trainable)
				self.conv10, self.w['conv10_w'], self.w['conv10_b'] = self.convolution_layer(self.conv9, layer_num = 10, weight_shape = [4, 4, self.layer_design[8], self.layer_design[9]], conv_strides = [1,2,2,1], trainable = trainable)
				self.conv11, self.w['conv11_w'], self.w['conv11_b'] = self.convolution_layer(self.conv10, layer_num = 11, weight_shape = [4, 4, self.layer_design[9], self.layer_design[10]], conv_strides = [1,2,2,1], trainable = trainable)
				self.conv12, self.w['conv12_w'], self.w['conv12_b'] = self.convolution_laeyr(self.conv11, layer_num = 12, weight_shape = [3, 3, self.layer_desigh[10], self.layer_design[11]], conv_strides = [1,1,1,1], trainable = trainable)
				
				# flatten layer
				layer_shape = self.conv3.get_shape()
				num_features = layer_shape[1:4].num_elements()
				self.flatten = tf.reshape(self.conv3, [-1, num_features]) #self.flatten_layer = tf.layers.flatten(self.conv3)

				# Dueling, Noisy layers
				#if flag == "noisy":
				#	with tf.variable_scope("", reuse = tf.AUTO_REUSE)
				#		self.value_fc, self.w['value_fc_w_mu'], self.w['value_fc_w_sigma'], self.w['value_fc_b_mu'], self.w['value_fc_b_sigma'] = noisy_dense(self.flatten, in_features = num_features, out_features = 512, name = "value_fc", trainable = tainable, activation_fn = tf.nn.relu)
				#		self.value, self.w['value_w_mu'], self.w['value_w_sigma'], self.w['value_b_mu'], self.w['value_b_sigma'] = noisy_dense(self.value_fc, in_features = 512, out_features = 1, name = "value", trainable = tainable, activation_fn = tf.identity)
				#		self.adv_fc, self.w['adv_fc_w_mu'], self.w['adv_fc_w_sigma'], self.w['adv_fc_b_mu'], self.w['adv_fc_b_sigma'] = noisy_dense(self.flatten, in_features = num_features, out_features = 512, name = "advantage_fc", trainable = trainable, activation_fn = tf.nn.relu)
				#		self.adv, self.w['adv_w_mu'], self.w['adv_w_sigma'], self.w['adv_b_mu'], self.w['adv_b_sigma'] = noisy_dense(self.adv_fc, in_features = 512, out_features = self.action_size, name = "advantage", trainable = trainable, activation_fn = tf.identity)
				#		self.output = self.value + tf.subtract(self.adv, tf.reduce_mean(self.adv, axis = 1, keepdims = True))

				# Dueling, Noisy, Distributional layers 
				#if flag == "all":
				with tf.variable_scope("", reuse = tf.AUTO_REUSE)
					self.value_fc, self.w['value_fc_w_mu'], self.w['value_fc_w_sigma'], self.w['value_fc_b_mu'], self.w['value_fc_b_sigma'] = noisy_dense(self.flatten, in_features = num_features, out_features = 512, name = "value_fc", trainable = tainable, activation_fn = tf.nn.relu)
					self.value, self.w['value_w_mu'], self.w['value_w_sigma'], self.w['value_b_mu'], self.w['value_b_sigma'] = noisy_dense(self.value_fc, in_features = 512, out_features = self.atoms, name = "value", trainable = tainable, activation_fn = tf.identity)
					self.value = tf.reshape(self.value, [-1, 1, self.atoms])
				
					self.adv_fc, self.w['adv_fc_w_mu'], self.w['adv_fc_w_sigma'], self.w['adv_fc_b_mu'], self.w['adv_fc_b_sigma'] = noisy_dense(self.flatten, in_features = num_features, out_features = 512, name = "advantage_fc", trainable = trainable, activation_fn = tf.nn.relu)
					self.adv, self.w['adv_w_mu'], self.w['adv_w_sigma'], self.w['adv_b_mu'], self.w['adv_b_sigma'] = noisy_dense(self.adv_fc, in_features = 512, out_features = self.action_size * self.atoms, name = "advantage", trainable = trainable, activation_fn = tf.identity)	
					self.adv = tf.reshape(self.adv, [-1, self.action_size, self.atoms])
						
					self.output = self.value + tf.subtract(self.adv, tf.reduce_mean(self.adv, axis = 1, keepdims = True))
					if log:
						self.output = tf.nn.log_softmax(self.output, axis = 2)
					else:
						self.output = tf.nn.softmax(self.output, axis = 2)

					self.output = tf.clip_by_value(self.output, 1e-8, 1.0-1e-8)

		def convolution_layer(self, input, layer_num, weight_shape, conv_strides = [1,4,4,1], trainable):
			with tf.variable_scope("conv" + str(layer_num), reuse = tf.AUTO_REUSE):
				w = tf.get_variable("w", shape = weight_shape, trainable = trainable, initializer = tf.contrib.layers.xavier_initializer())
				b = tf.get_variable("b", shape = weight_shape[-1], trainable = trainable, initializer = tf.zeros_initializer())
				conv_filter = tf.nn.conv2d(input, w, strides = conv_strides, padding = "VALID")
				L = tf.nn.relu(conv_filter + b)			
				#variable_summaries(w)
				#tf.summary.histogram("conv_filter", conv_filter)
				#tf.summary.histogram("relu_activation", L)
				#L = tf.nn.dropout(L, keep_prob = keep_prob)
				return L , w, b

		def noisy_dense(self, input, in_features, out_features, name, trainable, std_init = 0.4, activation_fn = tf.nn.relu):
			self.in_features = in_features
			self.out_features = out_features
			self.reset_noise()

			# initializer of mu and sigma
			_range = 1 / np.power(in_features, 0.5)
			mu_init = tf.random_uniform_initializer(minval = -1 * _range, maxval = 1 * _range)
			sigma_init = tf.constant_initializer(std_init / _range)

			# w = w_mu + w_sigma * w_epsilon
			w_mu = tf.get_variable(name + "/w_mu", [in_features, out_features], initializer=mu_init, trainable=trainable)
			w_sigma = tf.get_variable(name + "/w_sigma", [input_size, output_size], initializer=sigma_init, trainable=trainable)
			w = w_mu + tf.multiply(w_sigma, self.w_epsilon)
			ret = tf.matmul(input, w)

			b_mu = tf.get_variable(name + "/b_mu", [out_features], initializer=mu_init, trainable=trainable)
    		b_sigma = tf.get_variable(name + "/b_sigma", [out_features], initializer=sigma_init, trainable=trainable)
    		b = b_mu + tf.multiply(b_sigma, self.b_epsilon)
    		return activation_fn(ret + b), w_mu, w_sigma, b_mu, b_sigma

    	def reset_noise(self):
    		def f(x):
				return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))

			# sample noise from gaussian
			p = tf.random_normal([in_features, 1])
			q = tf.random_normal([1, out_features])
			f_p = f(p); f_q = f(q)
			self.w_epsilon = f_p * f_q
			self.b_epsilon = tf.squeeze(f_q)		


