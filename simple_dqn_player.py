#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import game
import random
import datetime

OBSERVE = 50000
BATCH_SIZE = 32
ACTION_HISTORY_LENGTH = 3
LOG_INTERVAL = 1000
SAVE_INTERVAL = 10000
MAX_PDATA_LIST_SIZE = 1000
MAX_D_SIZE = 1000000
GAMMA = 0.99
C = 10000 # Q reset interval

def phi(data):
	return (data == 0xc84848ff)

def conv2d(x, W, s):
  return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='VALID')

def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

# this function is using an optimized Q architecture of deepmind paper
def Q(state, keep_prob):
	with tf.name_scope('input'):
		# phi() -> 50x50x2
		#x = tf.placeholder(tf.float32, shape=[None, 50, 50, ACTION_HISTORY_LENGTH], name='x')
		pass

	with tf.name_scope('conv1'):
		# conv2d(x, W, 4) -> 11x11
		w_conv1 = weight_variable([8, 8, ACTION_HISTORY_LENGTH, 32], name='w_conv1')
		b_conv1 = bias_variable([32], name='b_conv1')
		h_conv1 = tf.nn.relu(conv2d(state, w_conv1, 4) + b_conv1)

	with tf.name_scope('conv2'):
		# conv2d(x, W, 2) -> 4x4
		w_conv2 = weight_variable([4, 4, 32, 64], name='w_conv2')
		b_conv2 = bias_variable([64], name='b_conv2')
		h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2, 2) + b_conv2)

	with tf.name_scope('conv3'):
		# conv2d(x, W, 1) -> 2x2
		w_conv3 = weight_variable([3, 3, 64, 64], name='w_conv3')
		b_conv3 = bias_variable([64], name='b_conv3')
		h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3, 1) + b_conv3)
	
	with tf.name_scope('flatten'):
		h_conv3_flat = tf.reshape(h_conv3, [-1, 2*2*64])

	with tf.name_scope('fc1'):
		# fully1
		W1 = weight_variable([2*2*64, 512], name='w_fc1')
		b1 = bias_variable([512], name='b_fc1')
		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W1)+b1)

	# dropout
	#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	h_fc1_drop = h_fc1

	with tf.name_scope('fc2'):
		# fully2
		W2 = weight_variable([512, 3], name='w_fc2')
		b2 = bias_variable([3], name='b_fc2')
		#fc2 = tf.nn.softmax(tf.matmul(h_fc1_drop,W2)+b2)
		fc2 = tf.matmul(h_fc1_drop,W2)+b2

	return fc2, (w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, W1, b1, W2, b2)


# pi(s) = argmax_a Q(s, a)
def pi(q_values):
	with tf.name_scope('pi'):
		readout = tf.argmax(q_values, 1)

	return readout

def max_Q(q_values):
	with tf.name_scope('max_q'):
		max_q = tf.reduce_max(q_values, reduction_indices=1)

	return max_q

def L(yj, a, q_values): # state = (s, a, r, s') -> (q, r, s')
	# 1/2 [r + max_a' Q(s', a') - Q(s, a)]**2
	with tf.name_scope('loss'):
		loss = tf.reduce_mean(tf.square(yj - tf.reduce_sum(q_values*tf.cast(a, tf.float32), reduction_indices=1)))
		
	return loss

def train(loss):
	#RMSProp
	with tf.name_scope('train'):
		train_op = tf.train.RMSPropOptimizer(0.00025,0.99,.0,1e-6).minimize(loss)

	return train_op

linear_controller = game.Controller()

with tf.Graph().as_default() as g:
	#model
	step = tf.Variable(0, name='step', trainable=False)
	ph_new_step = tf.placeholder(tf.int32, shape=[], name='new_step')
	assign_step = tf.assign(step, ph_new_step)

	ph_state = tf.placeholder(tf.float32, shape=[None, ACTION_HISTORY_LENGTH, 50, 50], name='state')
	ph_state_wrapper = tf.transpose(ph_state, perm=[0, 2, 3, 1]) # wrapper
	#ph_state = tf.placeholder(tf.float32, shape=[None, 50, 50, ACTION_HISTORY_LENGTH], name='state')
	#keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	keep_prob = None

	with tf.name_scope('input_summary'):
		input_summary=tf.image_summary('pdata', tf.reshape(tf.transpose(ph_state_wrapper[0:1,:,:,:], perm=[3,1,2,0]), [-1, 50, 50, 1]))

	q_values, theta = Q(ph_state_wrapper, keep_prob)
	action = pi(q_values)
	
	target_q_values, theta_m1 = Q(ph_state_wrapper, keep_prob)
	max_q = max_Q(target_q_values)

	with tf.name_scope('reset_target_q'):
		reset_target_q = list(tf.assign(lvalue, rvalue) for lvalue, rvalue in zip(theta_m1, theta))

	# for evaluating yj values
	with tf.name_scope('gamma_max_q'):
		gamma_max_q = GAMMA * max_q		

	ph_yj = tf.placeholder(tf.float32, shape=[None], name='yj') # yj = rj or rj + gamma * max_Q^(s, a)
	ph_a = tf.placeholder(tf.int64, shape=[None], name='a') # actions (3 selections, one hot encoded)
	ph_a_wrapper = tf.one_hot(ph_a, 3, 1, 0)
	#ph_a = tf.placeholder(tf.int32, shape=[None, 3], name='a') # actions (3 selections, one hot encoded)

	loss = L(ph_yj, ph_a_wrapper, q_values)
	train_op = train(loss)

	# update every input()
	ph_avg_reward = tf.placeholder(tf.float32, shape=[], name='avg_reward')
	reward_summary = tf.scalar_summary('_reward', ph_avg_reward)

	# update at new_episode()
	ph_avg_score_per_episode = tf.placeholder(tf.float32, shape=[], name='avg_score_per_episode')
	score_per_episode_summary = tf.scalar_summary('_score_per_episode', ph_avg_score_per_episode)

	# update at train
	ph_avg_loss = tf.placeholder(tf.float32, shape=[], name='avg_loss')
	loss_summary = tf.scalar_summary('_loss', ph_avg_loss)

	# update at inference
	ph_avg_max_q_value = tf.placeholder(tf.float32, shape=[], name='avg_max_q_value')
	max_q_value_summary = tf.scalar_summary('_max_q_value', ph_avg_max_q_value)

	saver = tf.train.Saver()

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	g.finalize()

	summary_writer=tf.train.SummaryWriter('logdir', sess.graph)

	checkpoint = tf.train.get_checkpoint_state("saved_networks")

	if checkpoint and checkpoint.model_checkpoint_path:
		saver.restore(sess, checkpoint.model_checkpoint_path)
		print("Successfully loaded:", checkpoint.model_checkpoint_path)
	else:
		print("Could not find old network weights")
		import os
		assert os.path.isdir('saved_networks')

	class MyController(game.Controller):
		def __init__(self):
			self.linear_controller = linear_controller
			self.D = []
			self.cnt = sess.run(step)
			self.last_action = 0
			self.last_score = 0

			self.total_reward = 0

			self.total_max_q_value = 0
			self.cnt_max_q_value = 0

			self.total_loss = 0
			self.cnt_loss = 0

			self.cnt_reward = 0

			self.total_score_per_episode = 0
			self.cnt_score_per_episode = 0
			#sess.run(tf.initialize_all_variables())
			
			#self.a_queue = []
			self.initial_pdata_list = [[[False] * 50] * 50] * 3 # deepcopy is not required
			self.pdata_list = self.initial_pdata_list[:]

		def save_networks(self):
			sess.run(assign_step, feed_dict={ph_new_step: self.cnt})
			saver.save(sess, 'saved_networks/' + 'network' + '-dqn', global_step=self.cnt)
			print('[%s] Successfully saved networks'%datetime.datetime.now())

		def state(self, previous):
			#return np.array(self.pdata_list[-ACTION_HISTORY_LENGTH-1:-1] if previous else self.pdata_list[-ACTION_HISTORY_LENGTH:]).swapaxes(0,1).swapaxes(1,2)
			return self.pdata_list[-ACTION_HISTORY_LENGTH-1:-1] if previous else self.pdata_list[-ACTION_HISTORY_LENGTH:]

		def new_episode(self, pf=None, data=None, reward=-1):
			self.pdata_list.append(phi(data))
		
			#print('new episode / terminal reward:',reward)

			# terminal
			if len(self.pdata_list) > ACTION_HISTORY_LENGTH:
				self.D.append([self.state(previous=True), self.last_action, reward, self.state(previous=False), True]) # e_t=(phi(s_t-1), a_t-1, r_t-1, phi(s_t), terminal)
				
			self.cnt_score_per_episode += 1
			self.total_score_per_episode += self.last_score

			if self.cnt_score_per_episode == (LOG_INTERVAL/10):
				summary_writer.add_summary(sess.run(score_per_episode_summary, feed_dict={ph_avg_score_per_episode:self.total_score_per_episode/self.cnt_score_per_episode}), self.cnt)
				self.total_score_per_episode = 0
				self.cnt_score_per_episode = 0

			self.pdata_list = self.initial_pdata_list[:]
			self.last_action = 0
			self.last_score = 0


		def handle_keyboardinterrupt(self):
			print('Received KeyboardInterrupt - Saving networks ...')
			self.save_networks()
			
		def input(self, pf=None, data=None, **kwargs):
			if data is not None:
				self.cnt += 1

				pdata = phi(data)
				self.pdata_list.append(pdata)
				
				#if pf.score == 0:
				#	summary_writer.add_summary(sess.run(input_summary, feed_dict={ph_state:[self.pdata_list[1:]]}))
				#	summary_writer.flush()
				#	exit()

				reward = pf.score - self.last_score
				self.total_reward += reward
				self.cnt_reward += 1

				#if reward:
				#	pass#print('reward:',reward)

				if len(self.pdata_list) > ACTION_HISTORY_LENGTH:
					self.D.append([self.state(previous=True), self.last_action, reward, self.state(previous=False), False]) # e_t=(phi(s_t-1), a_t-1, r_t-1, phi(s_t), terminal)
					
					if len(self.pdata_list) > MAX_PDATA_LIST_SIZE:
						self.pdata_list = self.pdata_list[-ACTION_HISTORY_LENGTH:]

				if random.random() < .1 or len(self.pdata_list) < ACTION_HISTORY_LENGTH or len(self.D) < OBSERVE: # exploration probability
					#if len(self.a_queue) == 0:
					#	self.a_queue = [random.randint(0,2)] * 4
					#a_t = self.a_queue.pop()#self.linear_controller.input(pf, data, **kwargs)+1 #random.randint(0,2)
					#a_t = self.linear_controller.input(pf, data, **kwargs)+1
					a_t = random.randint(0,2)
				else:
					ops = [action, max_q]
					feed = {ph_state: [self.state(previous=False)]}

					if self.cnt_max_q_value == LOG_INTERVAL:
						feed[ph_avg_max_q_value] = self.total_max_q_value / self.cnt_max_q_value
						self.total_max_q_value = 0
						self.cnt_max_q_value = 0
						ops.extend([input_summary, max_q_value_summary])

					ret = sess.run(ops, feed_dict=feed)
					a_t, current_max_q = ret[0][0], ret[1][0]

					self.total_max_q_value += current_max_q
					self.cnt_max_q_value += 1

					for summary in ret[2:]:
						summary_writer.add_summary(summary, self.cnt)

				if len(self.D) > OBSERVE:
					# train
					train_data = random.sample(self.D, BATCH_SIZE)

					yj_batch = []
					action_batch = []
					nonterminal_indices = []
					state_batch = []
					nonterminal_state_p1 = []

					for i, (t_state, t_action, t_reward, t_state_p1, t_terminal) in enumerate(train_data):
						state_batch.append(t_state)
						action_batch.append(t_action)
						yj_batch.append(t_reward)

						if not t_terminal:
							nonterminal_indices.append(i)
							nonterminal_state_p1.append(t_state_p1)

					q_values = sess.run(gamma_max_q, feed_dict={ph_state: nonterminal_state_p1})

					for i, q_value in zip(nonterminal_indices, q_values):
						yj_batch[i] += q_value

					_, current_loss = sess.run([train_op, loss], feed_dict={ph_yj: yj_batch, ph_state: state_batch, ph_a: action_batch})

					self.cnt_loss += 1
					self.total_loss += current_loss

					if self.cnt_loss == LOG_INTERVAL:
						summary_writer.add_summary(sess.run(loss_summary, feed_dict={ph_avg_loss: self.total_loss/self.cnt_loss}), self.cnt)
						self.total_loss = 0
						self.cnt_loss = 0

					if self.cnt % SAVE_INTERVAL == 0:
						self.save_networks()

					if len(self.D) > MAX_D_SIZE:
						self.D = self.D[OBSERVE:]

				if self.cnt_reward == LOG_INTERVAL:
					summary_writer.add_summary(sess.run(reward_summary, feed_dict={ph_avg_reward: self.total_reward/self.cnt_reward}), self.cnt)
					self.total_reward = 0
					self.cnt_reward = 0

				if self.cnt % C == 0:
					sess.run(reset_target_q)

				#assert 0 <= a_t <= 2

				self.last_action = a_t
				self.last_score = pf.score
				return a_t - 1

			return None

	game.Controller = MyController

	pf = game.PlayField(game.Server)
	pf.run()