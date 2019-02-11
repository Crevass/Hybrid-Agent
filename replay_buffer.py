import numpy as np
import time
	
# store data per iteration
class Replay_Buffer(object):
	def __init__(self, iter_step, lstm_step, capacity, ac_shape, ob_shape, lstm_nn):
		self.cur_num = 0
		self.lstm_step = lstm_step
		#self.transition_capacity = capacity
		iteration_capacity = capacity // iter_step
		self.iteration_capacity = iteration_capacity
		self.iter_step = iter_step
		self.replay = {
			"obs": np.zeros(shape=[iteration_capacity, iter_step+1]+list(ob_shape), dtype=np.float32),
			"acs": np.zeros(shape=[iteration_capacity, iter_step]+list(ac_shape), dtype=np.float32),
			"rews": np.zeros(shape=[iteration_capacity, iter_step], dtype=np.float32),
			"c": np.zeros(shape=[iteration_capacity, iter_step+1, lstm_nn], dtype=np.float32),
			"h": np.zeros(shape=[iteration_capacity, iter_step+1, lstm_nn], dtype=np.float32),
			"mask": np.ones(shape=[iteration_capacity, iter_step+1], dtype=np.int16)
		}
		self.ac_shape = ac_shape
		self.ob_shape = ob_shape
		self.lstm_nn = lstm_nn
		self.pointer = 0

		self.obs_mean = np.zeros(shape=list(ob_shape), dtype=np.float32)
		self.obs_std = np.ones(shape=list(ob_shape), dtype=np.float32)
		self.r_mean = 0.0
		self.r_std = 1.0
		self.r_max = 1.0
		self.r_min = -1.0



	def save(self, **kwargs):
		assert kwargs["obs"].shape[0] == self.iter_step+1
		assert kwargs["mask"].shape[0] == self.iter_step+1
		assert kwargs["acs"].shape[0] == self.iter_step
		for key in self.replay.keys():
			self.replay[key][self.pointer] = kwargs[key]
		self.cur_num = min(self.cur_num + 1, self.iteration_capacity)
		self.pointer = (self.pointer + 1) % self.iteration_capacity

	def sample_recent(self, sample_size):
		assert sample_size <= self.cur_num
		self.update_filter()
		if self.pointer - sample_size < 0:
			return {
				"obs": self.filter_ob(np.concatenate((self.replay["obs"][self.pointer-sample_size:], self.replay["obs"][:self.pointer]), axis=0)),
				"acs": np.concatenate((self.replay["acs"][self.pointer-sample_size:], self.replay["acs"][:self.pointer]), axis=0),
				"rews": self.filter_r(np.concatenate((self.replay["rews"][self.pointer-sample_size:], self.replay["rews"][:self.pointer]), axis=0)),
				"c": np.concatenate((self.replay["c"][self.pointer-sample_size:], self.replay["c"][:self.pointer]), axis=0),
				"h": np.concatenate((self.replay["h"][self.pointer-sample_size:], self.replay["h"][:self.pointer]), axis=0),
				"mask": np.concatenate((self.replay["mask"][self.pointer-sample_size:], self.replay["mask"][:self.pointer]), axis=0)
			}
		else:
			return {
				"obs": self.filter_ob(self.replay["obs"][self.pointer - sample_size: self.pointer]),
				"acs": self.replay["acs"][self.pointer - sample_size: self.pointer],
				"rews": self.filter_r(self.replay["rews"][self.pointer- sample_size: self.pointer]),
				"c": self.replay["c"][self.pointer - sample_size: self.pointer],
				"h": self.replay["h"][self.pointer - sample_size: self.pointer],
				"mask": self.replay["mask"][self.pointer - sample_size: self.pointer]
			}

	def sample_iteration(self, sample_size):
		assert sample_size <= self.cur_num

		self.update_filter()

		index = np.arange(self.cur_num)
		np.random.shuffle(index)
		index = index[:sample_size]
	
		return {
			"obs": self.filter_ob(self.replay["obs"][index]),
			"acs": self.replay["acs"][index],
			"rews": self.filter_r(self.replay["rews"][index]),
			"c": self.replay["c"][index],
			"h": self.replay["h"][index],
			"mask": self.replay["mask"][index]
		}
	
	def sample_lstm(self, sample_size):
		
		self.update_filter()
		all_limit = self.cur_num * (self.iter_step // self.lstm_step)
		batch_per_iter = self.iter_step // self.lstm_step
		assert sample_size <= all_limit
		index = np.zeros(shape=[all_limit, 2], dtype=np.int16)
		for i in range(self.cur_num):
			for j in range(batch_per_iter):
				index[i * batch_per_iter + j] = [i, j * self.lstm_step]
		np.random.shuffle(index)
		index = index[:sample_size]
		obs = np.zeros(shape=[sample_size, self.lstm_step+1]+list(self.ob_shape), dtype=np.float32)
		acs = np.zeros(shape=[sample_size, self.lstm_step]+list(self.ac_shape), dtype=np.float32)
		rews = np.zeros(shape=[sample_size, self.lstm_step], dtype=np.float32)
		c = np.zeros(shape=[sample_size, self.lstm_nn], dtype=np.float32)
		h = np.zeros(shape=[sample_size, self.lstm_nn], dtype=np.float32)
		second_c = np.zeros(shape=[sample_size, self.lstm_nn], dtype=np.float32)
		second_h = np.zeros(shape=[sample_size, self.lstm_nn], dtype=np.float32)
		mask = np.ones(shape=[sample_size, self.lstm_step+1], dtype=np.int16)
		for i in range(sample_size):
			iter_index = index[i, 0]
			step_index = index[i, 1]
			obs[i] = self.replay["obs"][iter_index, step_index:step_index+self.lstm_step+1]
			acs[i] = self.replay["acs"][iter_index, step_index:step_index+self.lstm_step]
			rews[i] = self.replay["rews"][iter_index, step_index:step_index+self.lstm_step]
			c[i] = self.replay["c"][iter_index, step_index]
			h[i] = self.replay["h"][iter_index, step_index]
			second_c[i] = self.replay["c"][iter_index, step_index+1]
			second_h[i] = self.replay["h"][iter_index, step_index+1]
			mask[i] = self.replay["mask"][iter_index, step_index:step_index+self.lstm_step+1]
		return {
			"obs": self.filter_ob(obs),
			"acs": acs,
			"rews": self.filter_r(rews),
			"c": c,
			"h": h,
			"second_c": second_c,
			"second_h": second_h,
			"mask": mask
		}


	def update_filter(self, tau=0.9):
		self.obs_mean = tau * self.obs_mean + (1.0 - tau) * np.mean(self.replay["obs"][:self.cur_num], axis=(0, 1))
		self.obs_std = tau * self.obs_std + (1.0 - tau) * np.std(self.replay["obs"][:self.cur_num], axis=(0, 1))
		self.r_mean = tau * self.r_mean + (1.0 - tau) * np.mean(self.replay["rews"][:self.cur_num])
		self.r_std = tau * self.r_std + (1.0 - tau) * np.std(self.replay["rews"][:self.cur_num])
		self.r_max = max(self.r_max, np.amax(self.replay["rews"][:self.cur_num]))
		self.r_min = min(self.r_min, np.amin(self.replay["rews"][:self.cur_num]))

	def filter_ob(self, x):
		return (x - self.obs_mean) / (self.obs_std + 1e-7)
		#return [(ob - self.obs_mean)/(self.obs_std+1e-6) for ob in x]
	def filter_r(self, x):
		# min->-1, max->+1
		#w = 2.0 / (self.r_max - self.r_min + 1e-7)
		#b = (self.r_max + self.r_min) / (self.r_min - self.r_max)
		#return w * x + b

		# min->0, max->+1
		w = 1.0 / (self.r_max - self.r_min + 1e-7)
		b = -self.r_min / (self.r_max - self.r_min + 1e-7)
		return (w * x + b)

		# use mean and std
		#return (x -self.r_mean) / (self.r_std + 1e-7)

		# return raw reward
		#return x
	def save_replays(self, save_path):
		for key in self.replay.keys():
			np.save(save_path+"/"+key+".npy", self.replay[key])
		print("replay data saved in "+save_path)

	def restore_replays(self, save_path):
		for key in self.replay.keys():
			self.replay[key][:] = np.load(save_path+"/"+key+".npy")
		self.cur_num = self.iteration_capacity
		self.update_filter(tau=0.0)
		print("-------------------------------")
		print("replay data restored.")
		#print("ob_mean: %.2f" %self.obs_mean)
		#print("ob_std: %.2f" %self.obs_std)
		print("r_max: %.2f" %self.r_max)
		print("r_min: %.2f" %self.r_min)
		print("-------------------------------")
