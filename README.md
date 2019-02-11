# Hybrid-Agent
On/off-policy hybrid agent and algorithm with LSTM network and tensorflow.
A method of hybrid agent and training algorithm using both on-policy loss function and off-policy loss function, reference to DDPG(http://arxiv.org/abs/1509.02971) and DPPO(http://arxiv.org/abs/1707.06347).

Require tensorflow, openAI gym and mujoco to train the agent.

# Start Training
To start training a new agent, run **testrun.py**. Tune the parameters in this file as you like.
Tensorflow ckpt files will be saved in **tf_saver**, and replay buffer's data will be saved in replays.