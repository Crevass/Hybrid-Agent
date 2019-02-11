import numpy as np
import tensorflow as tf
import sys
import time
class normal_dist(object):
	def __init__(self, loc, logstd):
		self.mean = loc
		self.logstd = logstd
		self.std = tf.exp(logstd)
		self.out = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
	def neglogp(self, x):
		result = 0.5*tf.reduce_sum(tf.square((x-self.mean)/self.std), axis=-1, keepdims=True)\
		+ 0.5*np.log(2.0*np.pi)*tf.to_float(tf.shape(x)[-1])\
		+ tf.reduce_sum(self.logstd, axis=-1)
		return result


class Agent(object):
	def __init__(self, **kwargs):
		ac_shape = kwargs["ac_shape"]
		ob_shape = kwargs["ob_shape"]
		lstm_step = kwargs["lstm_step"]
		lstm_nn = kwargs["lstm_nn"]
		layer_nn = kwargs["layer_nn"]

		self.gamma = kwargs["gamma"]
		self.lam = kwargs["lam"]

		self.sess = kwargs['session']
		self.lstm_nn = lstm_nn
		self.lstm_step = lstm_step
		self.ob_shape = ob_shape
		self.ac_shape = ac_shape
		self.main_obs = tf.placeholder(dtype=tf.float32, shape=[None]+[lstm_step]+list(ob_shape), name="main_obs")
		self.target_obs = tf.placeholder(dtype=tf.float32, shape=[None]+[lstm_step]+list(ob_shape), name="target_obs")
		#self.main_ob = tf.placeholder(dtype=tf.float32, shape=[None]+list(ob_shape), name="main_ob")
		#self.target_ob = tf.placeholder(dtype=tf.float32, shape=[None]+list(ob_shape), name="target_ob")
		#self.goal = tf.placeholder(dtype=tf.float32, shape=[None]+[step_num]+list(ob_shape), name="goal")
		#self.goal_2 = tf.placeholder(dtype=tf.float32, shape=[None]+[step_num]+list(ob_shape), name="goal_2")
		self.main_acs = tf.placeholder(dtype=tf.float32, shape=[None]+[lstm_step]+list(ac_shape), name="main_acs")
		self.target_acs = tf.placeholder(dtype=tf.float32, shape=[None]+[lstm_step]+list(ac_shape), name="target_acs")
		self.main_mask = tf.placeholder(dtype=tf.float32, shape=[None]+[lstm_step], name="main_mask")
		self.target_mask = tf.placeholder(dtype=tf.float32, shape=[None]+[lstm_step], name="target_mask")
		self.rewards = tf.placeholder(dtype=tf.float32, shape=[None]+[lstm_step], name="rewards")
		self.reward_mask = tf.placeholder(dtype=tf.float32, shape=[None]+[lstm_step], name="reward_mask")

		self.main_c_init = tf.placeholder(dtype=tf.float32, shape=[None, lstm_nn], name="main_c_init")
		self.main_h_init = tf.placeholder(dtype=tf.float32, shape=[None, lstm_nn], name="main_h_init")
		self.target_c_init = tf.placeholder(dtype=tf.float32, shape=[None, lstm_nn], name="target_c_init")
		self.target_h_init = tf.placeholder(dtype=tf.float32, shape=[None, lstm_nn], name="target_h_init")

		self.single_target_obs = tf.placeholder(dtype=tf.float32, shape=[None]+list(ob_shape), name="single_target_obs")
		self.single_target_acs = tf.placeholder(dtype=tf.float32, shape=[None]+list(ac_shape), name="single_target_acs")
		self.single_target_h = tf.placeholder(dtype=tf.float32, shape=[None, lstm_nn], name="single_target_h")
		self.single_target_c = tf.placeholder(dtype=tf.float32, shape=[None, lstm_nn], name="single_target_c")
		self.single_target_mask = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="single_target_mask")
		
		self.single_main_obs = tf.placeholder(dtype=tf.float32, shape=[None]+list(ob_shape), name="single_main_obs")
		self.single_main_acs = tf.placeholder(dtype=tf.float32, shape=[None]+list(ac_shape), name="single_main_acs")
		self.single_main_h = tf.placeholder(dtype=tf.float32, shape=[None, lstm_nn], name="single_main_h")
		self.single_main_c = tf.placeholder(dtype=tf.float32, shape=[None, lstm_nn], name="single_main_c")
		self.single_main_mask = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="single_main_mask")

		# for on-policy training per iteration
		self.ppo_ac = tf.placeholder(dtype=tf.float32, shape=[None, lstm_step]+list(ac_shape), name="ppo_ac")
		self.adv = tf.placeholder(dtype=tf.float32, shape=[None, lstm_step], name="adv")
		self.target_v = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="target_v")

		# learning rate
		self.pi_lr = tf.placeholder(dtype=tf.float32, name="pi_learning_rate")
		self.q_lr = tf.placeholder(dtype=tf.float32, name="q_learning_rate")

		# target_network update tau
		self.tau = tf.placeholder(dtype=tf.float32, name="tau")


		self.main = self._create_network(
						self.main_obs,
						self.main_acs,
						lstm_nn,
						layer_nn,
						self.main_mask,
						self.main_c_init,
						self.main_h_init,
						"main"
						)
		self.target = self._create_network(
						self.target_obs,
						self.target_acs,
						lstm_nn,
						layer_nn,
						self.target_mask,
						self.target_c_init,
						self.target_h_init,
						"target"
						)
		self.single_target = self._create_single_model(
						self.single_target_obs,
						self.single_target_acs,
						lstm_nn,
						layer_nn,
						self.single_target_mask,
						self.single_target_c,
						self.single_target_h,
						"target"
						)
		self.single_main = self._create_single_model(
						self.single_main_obs,
						self.single_main_acs,
						lstm_nn,
						layer_nn,
						self.single_main_mask,
						self.single_main_c,
						self.single_main_h,
						"main"
						)
		gamma = self.gamma
		#c1 = 0.8
		#c2 = 0.2
		#c3 = 0.5
		clip_range = 0.2

################ off-policy training loss ######################
		# create loss function
		with tf.variable_scope("loss"):
			split_mask = tf.split(value=self.reward_mask, axis=1, num_or_size_splits=self.lstm_step, name="reward_mask_split")
			split_r = tf.split(value=self.rewards, axis=1, num_or_size_splits=self.lstm_step, name="reward_split")
			next_vals = self.target["Q_pitrain"]
			value = self.target["Q_pitrain"][-1]
			q_loss = 0
			with tf.variable_scope("q_loss"):
				# q_loss creation
				for i in reversed(range(lstm_step)):
					value = split_r[i] + (1.0 - split_mask[i]) * gamma * value
					predict_q = split_r[i] + (1.0 - split_mask[i]) * gamma * next_vals[i]
					target_q = 0.6 * value + 0.4 * predict_q
					q_loss += tf.square(self.main["Q_qtrain"][i] - target_q, name="q_loss_square"+str(i))
				q_loss = tf.reduce_mean(q_loss, name="q_loss_reduce_mean")
				
			#pi_loss = 0
			with tf.variable_scope("pi_loss"):
				# pi_loss creation
				pi_loss = self.main["Q_pitrain"][-1]
				pi_loss = -tf.reduce_mean(pi_loss, name="pi_loss_reduce_mean")

#############################################################
			
################### on-policy loss ##########################
			pg_loss = 0
			with tf.variable_scope("pg_loss"):
				advs = tf.split(value=self.adv, axis=1, num_or_size_splits=self.lstm_step, name="split_adv")
				ppo_a = [tf.squeeze(v, axis=1) for v in tf.split(value=self.ppo_ac, axis=1, num_or_size_splits=self.lstm_step, name="split_ppo_ac")]
				for i in range(lstm_step):
					old_p = self.target["pi_dis"][i].neglogp(ppo_a[i])
					new_p = self.main["pi_dis"][i].neglogp(ppo_a[i])
					ratio = tf.exp(tf.stop_gradient(old_p) - new_p)
					surr1 = advs[i] * ratio
					surr2 = advs[i] * tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range)
					pg_loss += tf.minimum(surr1, surr2)

				pg_loss = -tf.reduce_mean(pg_loss)
			
			with tf.variable_scope("vf_loss"):
				vf_loss = tf.reduce_mean(tf.square(self.single_main["Q_qtrain"] - self.target_v), name="vf_loss_mean")
##############################################################
			"""
######################### l2_loss ############################
			l2_loss_pi = 0
			for p in tf.get_collection("l2_loss_pi_params", scope="main"):
				l2_loss_pi += tf.nn.l2_loss(p)
			l2_loss_pi = 0.00001 * l2_loss_pi

			l2_loss_q = 0
			for p in tf.get_collection("l2_loss_q_params", scope="main"):
				l2_loss_q += tf.nn.l2_loss(p)
			l2_loss_q = 0.00001 * l2_loss_q
##############################################################
			"""
		pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr, name="pi_optimizer")
		q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_lr, name="q_optimizer")

		pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "main/pi")
		q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "main/Q")

		target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target")
		main_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "main")

		with tf.name_scope("update_target"):
			self.update_target = [old.assign(old*(1.0-self.tau) + new * self.tau) for old, new in zip(target_params, main_params)]

		#pi_loss += l2_loss_pi
		#pg_loss += l2_loss_pi
		#q_loss += l2_loss_q
		#vf_loss += l2_loss_q

		self.pi_train = pi_optimizer.minimize(loss=pi_loss, var_list=pi_params)
		self.q_train = q_optimizer.minimize(loss=q_loss, var_list=q_params)
		self.pg_train = pi_optimizer.minimize(loss=pg_loss, var_list=pi_params)
		self.vf_train = q_optimizer.minimize(loss=vf_loss, var_list=q_params)

		self.summary_q_loss = tf.summary.scalar("q_loss", q_loss, family="off_policy_loss")
		self.summary_pi_loss = tf.summary.scalar("pi_loss", pi_loss, family="off_policy_loss")
		self.summary_pg_loss = tf.summary.scalar("pg_loss", pg_loss, family="on_policy_loss")
		self.summary_vf_loss = tf.summary.scalar("vf_loss", vf_loss, family="on_policy_loss")
		#self.summary_l2_loss_pi = tf.summary.scalar("l2_loss_pi", l2_loss_pi, family="l2_loss")
		#self.summary_l2_loss_q = tf.summary.scalar("l2_loss_q", l2_loss_q, family="l2_loss")

		# show variables and network construction
		print("--------------------global variables-----------------------")
		for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
			print(p)
		print("--------------------target variables-----------------------")
		for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target"):
			print(p)
		print("---------------------main variables------------------------")
		for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="main"):
			print(p)
		print("--------------------l2 loss variables----------------------")
		for p in tf.get_collection("l2_loss_pi_params", scope="main"):
			print(p)
		for p in tf.get_collection("l2_loss_q_params",scope="main"):
			print(p)




	def on_policy_train(self, iter_step, n_env, batch_size, optim_epoch, batch, pi_lr=0.00002, q_lr=0.00005, tau=0.2):
		assert batch["obs"].shape == (n_env, iter_step+1) + self.ob_shape
		assert batch["acs"].shape == (n_env, iter_step) + self.ac_shape
		batch_num = n_env * (iter_step // self.lstm_step)
		on_epoch = optim_epoch
		"""
		obs = np.zeros(shape=[batch_num, self.lstm_step+1]+list(self.ob_shape), dtype=np.float32)
		acs = np.zeros(shape=[batch_num, self.lstm_step]+list(self.ac_shape), dtype=np.float32)
		h = np.zeros(shape=[batch_num, self.lstm_nn], dtype=np.float32)
		c = np.zeros(shape=[batch_num, self.lstm_nn], dtype=np.float32)
		second_h = np.zeros(shape=[batch_num, self.lstm_nn], dtype=np.float32)
		second_c = np.zeros(shape=[batch_num, self.lstm_nn], dtype=np.float32)
		final_h = np.zeros(shape=[batch_num, self.lstm_nn], dtype=np.float32)
		final_c = np.zeros(shape=[batch_num, self.lstm_nn], dtype=np.float32)
		mask = np.ones(shape=[batch_num, self.lstm_step+1], dtype=np.int16)
		rews = np.zeros(shape=[batch_num, self.lstm_step], dtype=np.float32)
		"""
		vs = np.zeros(shape=[n_env, iter_step+1, 1], dtype=np.float32)
		adv = np.zeros(shape=[n_env, iter_step], dtype=np.float32)
		target_v = np.zeros(shape=[n_env, iter_step], dtype=np.float32)
		
		for i in range(n_env):
			vs[i, :iter_step] = self.sess.run(self.single_target["Q_qtrain"],
				feed_dict={
					self.single_target_obs: batch["obs"][i, 0:iter_step],
					self.single_target_acs: batch["acs"][i]
				})


		vs[:, iter_step] = self.sess.run(self.target["Q_pitrain"][-1],
				feed_dict={
					self.target_obs: batch["obs"][:, iter_step-self.lstm_step+1:],
					self.target_c_init: batch["c"][:, iter_step-self.lstm_step+1],
					self.target_h_init: batch["h"][:, iter_step-self.lstm_step+1],
					self.target_mask: batch["mask"][:, iter_step-self.lstm_step+1:]
				})

		for i in range(n_env):
			last_adv = 0
			value = vs[i][-1]
			for j in reversed(range(iter_step)):
				delta = batch["rews"][i, j] + self.gamma * vs[i, j+1] * (1.0-batch["mask"][i, j+1]) - vs[i, j]
				last_adv = delta + self.gamma * self.lam * (1.0-batch["mask"][i, j+1]) * last_adv
				adv[i, j] = last_adv
				value = batch["rews"][i, j] + self.gamma * value * (1.0-batch["mask"][i, j+1])
				target_v[i, j] = value
		# adv process
		#adv = (adv - np.mean(adv, axis=1, keepdims=True)) / np.std(adv, axis=1, keepdims=True)
		adv = (adv - np.mean(adv)) / np.std(adv)
		adv = np.reshape(adv, [batch_num, self.lstm_step])
		obs = np.reshape(batch["obs"][:, 0:iter_step], [batch_num, self.lstm_step]+list(self.ob_shape))
		acs = np.reshape(batch["acs"], [batch_num, self.lstm_step]+list(self.ac_shape))
		c = np.reshape(batch["c"][:, 0:iter_step], [batch_num, self.lstm_step, self.lstm_nn])[:, 0]
		h = np.reshape(batch["h"][:, 0:iter_step], [batch_num, self.lstm_step, self.lstm_nn])[:, 0]
		mask = np.reshape(batch["mask"][:, 0:iter_step], [batch_num, self.lstm_step])

		"""
		for i in range(n_env):
			for j in range(0, (iter_step//self.lstm_step)*self.lstm_step, self.lstm_step):
				ind = i * (iter_step//self.lstm_step) + (j//self.lstm_step)
				obs[ind] = batch["obs"][i][j:j+self.lstm_step+1]
				acs[ind] = batch["acs"][i][j:j+self.lstm_step]
				h[ind] = batch["h"][i][j]
				c[ind] = batch["c"][i][j]
				second_h[ind] = batch["h"][i][j+1]
				second_c[ind] = batch["c"][i][j+1]
				final_h[ind] = batch["h"][i][j+self.lstm_step]
				final_c[ind] = batch["c"][i][j+self.lstm_step]
				mask[ind] = batch["mask"][i][j:j+self.lstm_step+1]
				rews[ind] = batch["rews"][i][j:j+self.lstm_step]

		limit = iter_step // self.lstm_step * self.lstm_step
		"""
		

		# on-policy Q train
		vf_loss_summary = self.sess.run(self.summary_vf_loss, feed_dict={
						self.target_v: np.reshape(target_v, [n_env*iter_step, 1]),
						self.single_main_obs: np.reshape(batch["obs"][:, 0:iter_step], [n_env*iter_step]+list(self.ob_shape)),
						self.single_main_acs: np.reshape(batch["acs"], [n_env*iter_step]+list(self.ac_shape))
					})


		index = np.arange(n_env*iter_step)
		for _ in range(on_epoch):
			np.random.shuffle(index)
			for start in range(0, n_env*iter_step, batch_size):
				end = start + batch_size
				batch_index = index[start:end]
				self.sess.run(self.vf_train, feed_dict={
						self.target_v: np.reshape(target_v, [n_env*iter_step, 1])[batch_index],
						self.single_main_obs: np.reshape(batch["obs"][:, 0:iter_step], [n_env*iter_step]+list(self.ob_shape))[batch_index],
						self.single_main_acs: np.reshape(batch["acs"], [n_env*iter_step]+list(self.ac_shape))[batch_index],
						self.q_lr: q_lr
					})

		# on-policy pi train
		pg_loss_summary = self.sess.run(self.summary_pg_loss, feed_dict={
						self.adv: adv,
						self.main_obs: obs,
						self.target_obs: obs,
						self.main_c_init: c,
						self.main_h_init: h,
						self.target_c_init: c,
						self.target_h_init: h,
						self.main_mask: mask,
						self.target_mask: mask,
						self.ppo_ac: acs
					})

		index = np.arange(batch_num)
		for _ in range(on_epoch):
			np.random.shuffle(index)
			for start in range(0, batch_num, batch_size):
				end = start + batch_size
				batch_index = index[start:end]
				self.sess.run(self.pg_train, feed_dict={
						self.adv: adv[batch_index],
						self.main_obs: obs[batch_index],
						self.target_obs: obs[batch_index],
						self.main_c_init: c[batch_index],
						self.main_h_init: h[batch_index],
						self.target_c_init: c[batch_index],
						self.target_h_init: h[batch_index],
						self.main_mask: mask[batch_index],
						self.target_mask: mask[batch_index],
						self.ppo_ac: acs[batch_index],
						self.pi_lr: pi_lr
					})

		self.sess.run(self.update_target, feed_dict={self.tau: tau})
		return vf_loss_summary, pg_loss_summary
		

	
	def iteration_train(self, iter_step, batch_size, optim_epoch=5, pi_lr=0.00002, q_lr=0.00005, tau=0.2, **batch):

		vs = self.sess.run(self.single_target["Q_pitrain"],
				feed_dict={
					self.single_target_obs: batch["obs"],
					self.single_target_c: batch["c"],
					self.single_target_h: batch["h"],
					self.single_target_mask: batch["mask"].reshape([iter_step+1, 1])
				})
		assert vs.shape == (iter_step+1, 1)
		vs = np.squeeze(vs)
		assert vs.shape == (iter_step+1,)
		value = vs[-1]

		adv = np.zeros(shape=[iter_step], dtype=np.float32)
		target_v = np.zeros(shape=[iter_step, 1], dtype=np.float32)

		last_adv = 0
		for i in reversed(range(iter_step)):
			# create adv
			delta = batch["rews"][i] + self.gamma * vs[i+1] * (1.0-batch["mask"][i+1]) - vs[i]
			last_adv = delta + self.gamma * self.lam * (1.0-batch["mask"][i+1]) * last_adv
			adv[i] = last_adv

			# create target_V
			value = batch["rews"][i] + self.gamma * value * (1.0-batch["mask"][i+1])
			target_v[i] = [value]

		adv = (adv - adv.mean()) / (adv.std() + 1e-7)

		batch_num = iter_step // self.lstm_step
		limit = batch_num * self.lstm_step

		adv = np.reshape(adv[:limit], [batch_num, self.lstm_step])
		obs = np.reshape(batch["obs"][:limit], [batch_num, self.lstm_step]+list(self.ob_shape))
		acs = np.reshape(batch["acs"][:limit], [batch_num, self.lstm_step]+list(self.ac_shape))
		h = batch["h"][np.arange(0, iter_step, self.lstm_step)]
		c = batch["c"][np.arange(0, iter_step, self.lstm_step)]
		mask = np.reshape(batch["mask"][:limit], [batch_num, self.lstm_step])
		assert h.shape == (batch_num, self.lstm_nn)

		vf_loss_summary = self.sess.run(self.summary_vf_loss, feed_dict={
						self.target_v: target_v,
						self.single_main_obs: batch["obs"][0:len(batch["acs"])],
						self.single_main_acs: batch["acs"]
					})

		index = np.arange(iter_step)
		for epoch in range(optim_epoch):
			np.random.shuffle(index)
			for start in range(0, iter_step, batch_size):
				end = start + batch_size
				batch_index = index[start:end]
				self.sess.run(self.vf_train, feed_dict={
						self.target_v: target_v[batch_index],
						self.single_main_obs: batch["obs"][batch_index],
						self.single_main_acs: batch["acs"][batch_index],
						self.q_lr: q_lr
					})

		pg_loss_summary = self.sess.run(self.summary_pg_loss, feed_dict={
						self.adv: adv,
						self.main_obs: obs,
						self.target_obs: obs,
						self.main_c_init: c,
						self.main_h_init: h,
						self.target_c_init: c,
						self.target_h_init: h,
						self.main_mask: mask,
						self.target_mask: mask,
						self.ppo_ac: acs
					})

		index = np.arange(batch_num)
		for epoch in range(optim_epoch):
			np.random.shuffle(index)
			for start in range(0, (batch_num//batch_size)*batch_size, batch_size):
				end = start + batch_size
				batch_index = index[start:end]

				self.sess.run(self.pg_train, feed_dict={
						self.adv: adv[batch_index],
						self.main_obs: obs[batch_index],
						self.target_obs: obs[batch_index],
						self.main_c_init: c[batch_index],
						self.main_h_init: h[batch_index],
						self.target_c_init: c[batch_index],
						self.target_h_init: h[batch_index],
						self.main_mask: mask[batch_index],
						self.target_mask: mask[batch_index],
						self.ppo_ac: acs[batch_index],
						self.pi_lr: pi_lr
					})

		self.sess.run(self.update_target, feed_dict={self.tau: tau})
		return vf_loss_summary, pg_loss_summary

	

	def off_policy_train(self, batch, sample_size, batch_size, optim_epoch=10, pi_lr=0.00002, q_lr=0.00005, tau=0.1):

		assert batch["obs"].shape == (sample_size, self.lstm_step+1) + self.ob_shape
		assert batch["acs"].shape == (sample_size, self.lstm_step) + self.ac_shape
		
		q_loss_summary = self.sess.run(self.summary_q_loss, feed_dict={
						self.main_obs: batch["obs"][:, 0:self.lstm_step],
						self.main_acs: batch["acs"],
						self.rewards: batch["rews"],
						self.reward_mask: batch["mask"][:, 1:],
						self.target_obs: batch["obs"][:, 1:],
						self.target_h_init: batch["second_h"],
						self.target_c_init: batch["second_c"],
						self.target_mask: batch["mask"][:, 1:]
					})
		"""
		q_loss_summary = self.sess.run(self.summary_q_loss, feed_dict={
						self.rewards: batch["rews"],
						self.reward_mask: batch["mask"][:, 1:],
						self.target_obs: batch["obs"][:, 1:],
						self.target_h_init: batch["second_h"],
						self.target_c_init: batch["second_c"],
						self.target_mask: batch["mask"][:, 1:],
						self.single_main_obs: batch["obs"][:, self.lstm_step-1],
						self.single_main_acs: batch["acs"][:, self.lstm_step-1]
					})
		"""

		pi_loss_summary = self.sess.run(self.summary_pi_loss, feed_dict={
						self.main_obs: batch["obs"][:, 0:self.lstm_step],
						self.main_c_init: batch["c"],
						self.main_h_init: batch["h"],
						self.main_mask: batch["mask"][:, 0:self.lstm_step]
			})

		index = np.arange(sample_size)
		for _ in range(optim_epoch):
			np.random.shuffle(index)
			for start in range(0, (sample_size//batch_size)*batch_size, batch_size):
				end = start + batch_size
				batch_index = index[start:end]
				
				self.sess.run(self.q_train, 
					feed_dict={
						self.main_obs: batch["obs"][batch_index, 0:self.lstm_step],
						self.main_acs: batch["acs"][batch_index],
						self.rewards: batch["rews"][batch_index],
						self.reward_mask: batch["mask"][batch_index, 1:],
						self.target_obs: batch["obs"][batch_index, 1:],
						self.target_h_init: batch["second_h"][batch_index],
						self.target_c_init: batch["second_c"][batch_index],
						self.target_mask: batch["mask"][batch_index, 1:],
						self.q_lr: q_lr
					})
				"""	
				self.sess.run(self.q_train, 
					feed_dict={
						self.rewards: batch["rews"][batch_index],
						self.reward_mask: batch["mask"][batch_index, 1:],
						self.target_obs: batch["obs"][batch_index, 1:],
						self.target_h_init: batch["second_h"][batch_index],
						self.target_c_init: batch["second_c"][batch_index],
						self.target_mask: batch["mask"][batch_index, 1:],
						self.single_main_obs: batch["obs"][batch_index, self.lstm_step-1],
						self.single_main_acs: batch["acs"][batch_index, self.lstm_step-1],
						self.q_lr: q_lr
					})
				"""
				self.sess.run(self.pi_train,
					feed_dict={
						self.main_obs: batch["obs"][batch_index, 0:self.lstm_step],
						self.main_c_init: batch["c"][batch_index],
						self.main_h_init: batch["h"][batch_index],
						self.main_mask: batch["mask"][batch_index, 0:self.lstm_step],
						self.pi_lr: pi_lr
					})
		#l2_loss_pi_summary = self.sess.run(self.summary_l2_loss_pi)
		#l2_loss_q_summary = self.sess.run(self.summary_l2_loss_q)

		self.sess.run(self.update_target, feed_dict={self.tau: tau})
		return q_loss_summary, pi_loss_summary


	def layer_norm(self, x, g, b, e=1e-7, axes=1):
		mean, var = tf.nn.moments(x, axes=axes, keep_dims=True)
		x = (x-mean)/tf.sqrt(var+e)
		x = x * g + b
		return x

	def _critic(self, xs, unit_nn, reuse=False):
		x_shape = xs[0].shape[-1]
		w1_shape = None if reuse else [x_shape, unit_nn]
		w2_shape = None if reuse else [unit_nn, unit_nn]
		w3_shape = None if reuse else [unit_nn, 1]

		b1_shape = None if reuse else [unit_nn]
		b2_shape = None if reuse else [unit_nn]
		#b3_shape = None if reuse else [1]

		init1 = None
		init2 = None if reuse else tf.constant_initializer(0.01)

		w1 = tf.get_variable("w1", shape=w1_shape, initializer=init1)
		w2 = tf.get_variable("w2", shape=w2_shape, initializer=init1)
		w3 = tf.get_variable("w3", shape=w3_shape, initializer=init1)

		b1 = tf.get_variable("b1", shape=b1_shape, initializer=init2)
		b2 = tf.get_variable("b2", shape=b2_shape, initializer=init2)
		#b3 = tf.get_variable("b3", shape=b3_shape, initializer=init2)

		if not reuse:
			tf.add_to_collection("l2_loss_q_params", w1)
			tf.add_to_collection("l2_loss_q_params", w2)
			tf.add_to_collection("l2_loss_q_params", w3)

		qs = []
		for x in xs:
			xx = tf.matmul(x, w1) + b1
			xx = tf.nn.leaky_relu(xx)
			xx = tf.matmul(xx, w2) + b2
			xx = tf.nn.leaky_relu(xx)
			q = tf.matmul(xx, w3)
			qs.append(q)
		return qs


	def _lstm(self, xs, out_shape, cell_nn, c_init, h_init, masks, reuse=False):
		# split input tensor into inputs of lstm network
		#xs = [tf.squeeze(v, axis=1) for v in tf.split(value=xs, axis=1, num_or_size_splits=self.step_num)]

		c = c_init
		h = h_init
		x_shape = xs[0].shape[-1]
		
		wx_shape = None if reuse else [x_shape, cell_nn*4]
		gx_shape = None if reuse else [cell_nn*4]
		bx_shape = None if reuse else [cell_nn*4]

		wh_shape = None if reuse else [cell_nn, cell_nn*4]
		gh_shape = None if reuse else [cell_nn*4]
		bh_shape = None if reuse else [cell_nn*4]

		b_shape = None if reuse else [cell_nn*4]

		gc_shape = None if reuse else [cell_nn]
		bc_shape = None if reuse else [cell_nn]

		w_readout_shape = None if reuse else [cell_nn]+list(out_shape)
		b_readout_shape = None if reuse else list(out_shape)
		gr_shape = None if reuse else list(out_shape)
		br_shape = None if reuse else list(out_shape)

		std_shape = None if reuse else list(out_shape)

		init1 = None if reuse else tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
		init2 = None if reuse else tf.constant_initializer(0.01)
		init3 = None if reuse else tf.constant_initializer(0.0001)
		init4 = None if reuse else tf.constant_initializer(1.0)

		wx = tf.get_variable("wx", shape=wx_shape, initializer=init1)
		gx = tf.get_variable("gx", shape=gx_shape, initializer=init4)
		bx = tf.get_variable("bx", shape=bx_shape, initializer=init3)

		wh = tf.get_variable("wh", shape=wh_shape, initializer=init1)
		gh = tf.get_variable("gh", shape=gh_shape, initializer=init4)
		bh = tf.get_variable("bh", shape=bh_shape, initializer=init3)

		b = tf.get_variable("b", shape=b_shape, initializer=init2)

		gc = tf.get_variable("gc", shape=gc_shape, initializer=init4)
		bc = tf.get_variable("bc", shape=bc_shape, initializer=init3)

		w_readout = tf.get_variable("w_readout", shape=w_readout_shape, initializer=init1)
		#b_readout = tf.get_variable("b_readout", shape=b_readout_shape, initializer=init2)
		#g_readout = tf.get_variable("g_readout", shape=gr_shape, initializer=init4)
		#b_readout = tf.get_variable("b_readout", shape=br_shape, initializer=init3)
		logstd = tf.get_variable("logstd", shape=std_shape, initializer=tf.constant_initializer(0.3))

		if not reuse:
			tf.add_to_collection("l2_loss_pi_params", wx)
			tf.add_to_collection("l2_loss_pi_params", wh)
			tf.add_to_collection("l2_loss_pi_params", w_readout)

		hs, cs, outs, distributions = [], [], [], []
		#hs, cs, outs = [], [], []

		### implement LN before split or after split? ###

		for index, (x, m) in enumerate(zip(xs, masks)):
			with tf.name_scope("lstm_cell_"+str(index)):
				c = c*(1.0-m)
				h = h*(1.0-m)

				z = self.layer_norm(tf.matmul(x, wx), gx, bx) + self.layer_norm(tf.matmul(h, wh), gh, bh) + b

				f, i, u, o = tf.split(value=z, axis=1, num_or_size_splits=4)

				# forget gate
				f = tf.nn.sigmoid(f)
				# input gate
				i = tf.nn.sigmoid(i)
				# new update info
				u = tf.tanh(u)
				# new o
				o = tf.nn.sigmoid(o)
				with tf.name_scope("cell_status_"+str(index)):
					# new cell status
					c = f * c + i * u
				with tf.name_scope("h_"+str(index)):
					# new h
					h = o * tf.tanh(self.layer_norm(c, gc, bc))
				# new readout
				readout = tf.matmul(h, w_readout)
				#readout = self.layer_norm(tf.matmul(h, w_readout), g_readout, b_readout)
				with tf.name_scope("readout_"+str(index)):
					readout = tf.tanh(readout, name="readout_tanh_"+str(index))
				with tf.name_scope("out_distribution"):
					normal = normal_dist(readout, logstd)

				hs.append(h)
				cs.append(c)
				outs.append(readout)
				distributions.append(normal)

		return hs, cs, outs, distributions

	def step(self, x, c, h, mask, mode="train", prob=0.0):
		assert mode in ["train", "test"]
		length = len(x)
		feed_o = np.zeros(shape=[self.lstm_step]+list(self.ob_shape), dtype=np.float32)
		feed_m = np.zeros(shape=[self.lstm_step], dtype=np.float32)
		for i in range(length):
			feed_o[i] = x[i]
			feed_m[i] = mask[i]
		feed_o = feed_o[np.newaxis, :]
		feed_m = feed_m[np.newaxis, :]
		c = c[np.newaxis, :]
		h = h[np.newaxis, :]
		fdict = {
			self.target_obs: feed_o,
			self.target_h_init: h,
			self.target_c_init: c,
			self.target_mask: feed_m
		}

		if mode == "train":
			final_out, final_h, final_c = self.sess.run(
				[
				self.target["pi_dis"][length-1].out,
				self.target["pi_hs"][length-1],
				self.target["pi_cs"][length-1]
				],
				feed_dict=fdict)
			if np.random.choice([0, 1], p=[1.0 - prob, prob]) == 1:
				final_out = np.random.uniform(low=-1.0, high=1.0, size=final_out.shape)

		elif mode == "test":
			final_out, final_h, final_c = self.sess.run(
				[
				self.target["pi_outs"][length-1],
				self.target["pi_hs"][length-1],
				self.target["pi_cs"][length-1]
				],
				feed_dict=fdict)

		final_out = np.squeeze(final_out, axis=0)
		final_out = np.clip(final_out, -1.0, 1.0)
		final_h = np.squeeze(final_h, axis=0)
		final_c = np.squeeze(final_c, axis=0)
		return final_out, final_h, final_c

	def _create_network(self, obs, acs, cell_nn, critic_nn, masks, c_init, h_init, scope):
		with tf.variable_scope(scope):
			split_obs = [tf.squeeze(v, axis=1) for v in tf.split(value=obs, axis=1, num_or_size_splits=self.lstm_step, name=scope+"_split_obs")]
			split_acs = [tf.squeeze(v, axis=1) for v in tf.split(value=acs, axis=1, num_or_size_splits=self.lstm_step, name=scope+"_split_acs")]
			#split_mask = [tf.squeeze(v, axis=1) for v in tf.split(value=masks, axis=1, num_or_size_splits=self.lstm_step)]
			split_mask = tf.split(value=masks, axis=1, num_or_size_splits=self.lstm_step, name=scope+"_split_mask")
			with tf.variable_scope("pi"):
				pi_hs, pi_cs, pi_outs, pi_dis = self._lstm(split_obs, self.ac_shape, cell_nn, c_init, h_init, split_mask)
			with tf.name_scope("Q_pitrain_input"):
				Q_pitrain_input = [tf.concat([a.out, o], axis=-1) for a, o in zip(pi_dis, split_obs)]
			with tf.name_scope("Q_qtrain_input"):
				Q_qtrain_input = [tf.concat([a, o], axis=-1) for a, o in zip(split_acs, split_obs)]
			with tf.variable_scope("Q"):
				Q_pitrain = self._critic(Q_pitrain_input, critic_nn)
			with tf.variable_scope("Q", reuse=True):
				Q_qtrain = self._critic(Q_qtrain_input, critic_nn, reuse=True)
		res = {
		"pi_hs": pi_hs,
		"pi_cs": pi_cs,
		"pi_outs": pi_outs,
		"pi_dis": pi_dis,
		"Q_pitrain": Q_pitrain,
		"Q_qtrain": Q_qtrain
		}
		return res
	def _create_single_model(self, ob, ac, cell_nn, critic_nn, mask, c_init, h_init, scope):
		with tf.variable_scope(scope):
			with tf.variable_scope("pi", reuse=True):
				h, c, out, dis = self._lstm([ob], self.ac_shape, cell_nn, c_init, h_init, [mask], reuse=True)
			assert len(out) == 1
			h, c, out, dis = h[0], c[0], out[0], dis[0]
			inpt = tf.concat([dis.out, ob], axis=-1)
			inpt2 = tf.concat([ac, ob], axis=-1)
			with tf.variable_scope("Q", reuse=True):
				Q = self._critic([inpt], critic_nn, reuse=True)
			assert len(Q) == 1
			Q = Q[0]
			with tf.variable_scope("Q", reuse=True):
				Q_qtrain = self._critic([inpt2], critic_nn, reuse=True)
			assert len(Q_qtrain) == 1
			Q_qtrain = Q_qtrain[0]

		res = {
			"pi_hs": h,
			"pi_cs": c,
			"pi_outs": out,
			"pi_dis": dis,
			"Q_pitrain": Q,
			"Q_qtrain": Q_qtrain
		}
		return res