import tensorflow as tf
import numpy as np
import gym
from replay_buffer import Replay_Buffer
from agent import Agent
import time

# ENVS

#ENV_NAME = "Pendulum-v0"
#ENV_NAME = "LunarLanderContinuous-v2"
#ENV_NAME = "Humanoid-v2"
#ENV_NAME = "my_Ant-v0"
#ENV_NAME = "FetchPickAndPlace-v1"
#ENV_NAME = "HumanoidStandup-v2"
ENV_NAME = "my_Humanoid-v0"


env = gym.make(ENV_NAME)
ob_shape = env.observation_space.shape
ac_shape = env.action_space.shape
ac_high = env.action_space.high
ac_low = env.action_space.low
env.close()


# define network unit number
LSTM_NN = 128
LSTM_STEP = 8
DENSE_NN = 128

# define iteration limit
MAX_ITER = 15000
STEP_PER_ITER = 1024

# define training parameters
ON_ENVS = 4
OFF_ENVS = 1
N_ENVS = ON_ENVS + OFF_ENVS
BATCH_SIZE = 32
OPTIM_EPOCH = 5
OFF_SAMPLE_SIZE = N_ENVS * STEP_PER_ITER  // LSTM_STEP
ON_SAMPLE_SIZE = ON_ENVS
#REPLAY_CAPACITY = 2500000
REPLAY_CAPACITY = STEP_PER_ITER * 1500
SAMPLE_HORIZON = 10
RECORD_TURN = 50

UPDATE_TAU = 0.2
#assert SAMPLE_SIZE*(STEP_PER_ITER//LSTM_STEP) > BATCH_SIZE

# file operation
Restore_Iter = 12749
START_ITER = 0


def record_video(model, env, replay, test_turn=3):
	obs = np.zeros(shape=[LSTM_STEP]+list(ob_shape), dtype=np.float32)
	hs = np.zeros(shape=[LSTM_NN], dtype=np.float32)
	cs = np.zeros(shape=[LSTM_NN], dtype=np.float32)
	mask = np.ones(shape=[LSTM_STEP], dtype=np.int16)
	for i in range(test_turn):
		obs[0] = env.reset()
		hs = np.zeros(shape=[LSTM_NN], dtype=np.float32)
		cs = np.zeros(shape=[LSTM_NN], dtype=np.float32)
		mask[0] = 1
		pointer = 0
		while True:
			out, h, c = model.step(replay.filter_ob(obs[:pointer+1]), hs, cs, mask[:pointer+1], mode="test")
			a = out*ac_high
			nextob, r, done, _ = env.step(a)
			pointer = (pointer+1)%LSTM_STEP
			if pointer == 0:
				hs = h
				cs = c
			obs[pointer] = nextob
			mask[pointer] = done
			if done:
				break
"""
def eval(model, env, replay, test_turn=3):
	res = []
	obs = np.zeros(shape=[LSTM_STEP]+list(ob_shape), dtype=np.float32)
	hs = np.zeros(shape=[LSTM_NN], dtype=np.float32)
	cs = np.zeros(shape=[LSTM_NN], dtype=np.float32)
	mask = np.ones(shape=[LSTM_STEP], dtype=np.int16)
	for i in range(test_turn):
		obs[0] = env.reset()
		hs = np.zeros(shape=[LSTM_NN], dtype=np.float32)
		cs = np.zeros(shape=[LSTM_NN], dtype=np.float32)
		mask[0] = 1
		pointer = 0
		ep_reward = 0
		while True:
			out, h, c = model.step(replay.filter_ob(obs[:pointer+1]), hs, cs, mask[:pointer+1], mode="test")
			a = out*ac_high
			nextob, r, done, _ = env.step(a)
			ep_reward += r
			pointer = (pointer+1)%LSTM_STEP
			if pointer == 0:
				hs = h
				cs = c
			obs[pointer] = nextob
			mask[pointer] = done
			if done:
				res.append(ep_reward)
				break
	return np.mean(res)
"""


def main():
	env = gym.make(ENV_NAME)
	test_env = gym.make(ENV_NAME)
	test_env = gym.wrappers.Monitor(
			env=test_env,
			directory="./video/",
			video_callable=lambda x:True,
			force=True,
			mode='evaluation')



	gpu_option = tf.GPUOptions(
				#per_process_gpu_memory_fraction=0.4,
				allow_growth=True
				)
	sess = tf.Session(
				config=tf.ConfigProto(
					gpu_options=gpu_option,
					log_device_placement=False
					)
				)

	model = Agent(
				ac_shape=ac_shape,
				ob_shape=ob_shape, 
				lstm_step=LSTM_STEP,
				lstm_nn=LSTM_NN,
				layer_nn=DENSE_NN,
				gamma=0.95,
				lam=0.95,
				session=sess,)

	reward_record = tf.placeholder(dtype=tf.float32, shape=(), name="reward_record")
	summary_reward = tf.summary.scalar("reward", reward_record)

	replay = Replay_Buffer(STEP_PER_ITER, LSTM_STEP, REPLAY_CAPACITY, ac_shape, ob_shape, LSTM_NN)
	writer = tf.summary.FileWriter("./tf_log/", sess.graph)

	#sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.global_variables())

	###### restore parameters ###################
	if Restore_Iter:
		saver.restore(sess, "./tf_saver/save_params-"+str(Restore_Iter))
		print("network params restored")
	#############################################


	#global_step = 0
	epreward = 0
	states = np.zeros(shape=[STEP_PER_ITER+1]+list(ob_shape), dtype=np.float32)
	actions = np.zeros(shape=[STEP_PER_ITER]+list(ac_shape), dtype=np.float32)
	rews = np.zeros(shape=[STEP_PER_ITER], dtype=np.float32)
	hs = np.zeros(shape=[STEP_PER_ITER+1, LSTM_NN], dtype=np.float32)
	cs = np.zeros(shape=[STEP_PER_ITER+1, LSTM_NN], dtype=np.float32)
	mask = np.ones(shape=[STEP_PER_ITER+1], dtype=np.int16)

	record = np.zeros(shape=[MAX_ITER], dtype=np.float32)

	# load record array
	"""
	record[:] = np.load("rewards.npy")[:MAX_ITER]
	assert record[Restore_Iter] != 0
	assert record[START_ITER] == 0
	print("record restored")
	replay.restore_replays("./replays")
	"""

	# initialize env
	states[0] = env.reset()

	recent_rewards = []
	for iters in range(START_ITER, MAX_ITER):
		time_start = time.time()

		pi_lr = np.round(0.00004 - iters*(0.00004-0.00001)/MAX_ITER, 7)
		q_lr = np.round(0.00008 - iters*(0.00008-0.00001)/MAX_ITER, 7)
		n_worker = iters % N_ENVS
		prob = 0.1 if n_worker < OFF_ENVS else 0.0
		
		print("---------------------------")
		print("iters: %i, worker: %i" %(iters, n_worker))
		print("pi_lr: %.6f, q_lr: %.6f" %(pi_lr, q_lr))
		print("random action prob: %.2f" %prob)

		for step in range(STEP_PER_ITER):
			# slice input with step
			start_index = max(0, step - LSTM_STEP + 1)

			out, pi_h, pi_c = model.step(
				replay.filter_ob(states[start_index:step+1]), 
				cs[start_index], 
				hs[start_index], 
				mask[start_index:step+1],
				mode="train",
				prob=prob)
			
			a = out*ac_high
			
			ob_next, rew, done, _ = env.step(a)
			epreward += rew
			if done:
				ob_next = env.reset()
				if len(recent_rewards) >= 100:
					recent_rewards.pop(0)
				recent_rewards.append(epreward)
				epreward = 0
			
			states[step+1] = ob_next
			cs[step+1] = pi_c
			hs[step+1] = pi_h
			mask[step+1] = done
			actions[step] = out
			rews[step] = rew

			#global_step += 1

		# store iteration's trainsition
		replay.save(obs=states, acs=actions, rews=rews, c=cs, h=hs, mask=mask)

		# train model

		# iteration train on model
		s1, s2 = model.iteration_train(
						STEP_PER_ITER,
						BATCH_SIZE,
						OPTIM_EPOCH,
						pi_lr=pi_lr,
						q_lr=q_lr,
						tau=UPDATE_TAU,
						obs=replay.filter_ob(states),
						acs=actions,
						rews=replay.filter_r(rews),
						c=cs,
						h=hs,
						mask=mask)
		writer.add_summary(s1, global_step=iters)
		writer.add_summary(s2, global_step=iters)

		if (replay.cur_num >= SAMPLE_HORIZON) and ((iters + 1) % N_ENVS == 0):
			"""
			batch = replay.sample_recent(ON_SAMPLE_SIZE)
			s1, s2 = model.on_policy_train(
					STEP_PER_ITER,
					ON_SAMPLE_SIZE,
					BATCH_SIZE,
					OPTIM_EPOCH,
					batch,
					pi_lr=pi_lr,
					q_lr=q_lr)
			"""
			batch = replay.sample_lstm(OFF_SAMPLE_SIZE)
			s3, s4 = model.off_policy_train(
					batch,
					OFF_SAMPLE_SIZE,
					BATCH_SIZE,
					OPTIM_EPOCH,
					pi_lr=pi_lr,
					q_lr=q_lr,
					tau=UPDATE_TAU)
			writer.add_summary(s3, global_step=iters)
			writer.add_summary(s4, global_step=iters)
			#writer.add_summary(s5, global_step=iters)
			#writer.add_summary(s6, global_step=iters)
		# clean up for next iteration
		states[0] = states[-1]
		cs[0] = cs[-1]
		hs[0] = hs[-1]
		mask[0] = mask[-1]

		record[iters] = np.mean(recent_rewards)
		s7 = sess.run(summary_reward, feed_dict={reward_record: np.mean(recent_rewards)})
		writer.add_summary(s7, global_step=iters)
		time_end = time.time()
		t = time_end - time_start
		t_s = t // 1
		t_ms = np.round((t - t_s) * 1000)
		print("   --> avg: %.3f, memo: %i" %(np.mean(recent_rewards), replay.cur_num))
		print("   --> time: %is %ims" %(t_s, t_ms))

		# record iteration
		if (iters+1) % RECORD_TURN == 0:
			print("Iteration: %i, start testing." %iters)
			record_video(model, test_env, replay)
			saver.save(sess, "./tf_saver/save_params", global_step=iters)
			np.save("rewards.npy", record)
			print("Rewards saved in reward.npy")


if __name__ == "__main__":
	main()