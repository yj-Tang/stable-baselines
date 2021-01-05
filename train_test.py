# import gym
# import numpy as np
#
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines import SAC
#
# # env = gym.make('Pendulum-v0')
# env = gym.make('Ex3_EKF_gyro-v0')
#
# model = SAC(MlpPolicy, env, verbose=1, tensorboard_log="./logs/")
# # model.learn(total_timesteps=1000000, log_interval=10, tb_log_name="sac_Pendulum_2")
# model.learn(total_timesteps=3000000, log_interval=10, tb_log_name="sac_ekf_3")
# # model.save("sac_pendulum_model_2")
# model.save("sac_ekf_model_3")


# import gym
# import numpy as np
#
# from stable_baselines.ddpg.policies import MlpPolicy
# from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
# from stable_baselines import DDPG
#
# env = gym.make('MountainCarContinuous-v0')
#
# # the noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# param_noise = None
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
#
# model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
# model.learn(total_timesteps=400000)
# model.save("ddpg_mountain")
#
# del model # remove to demonstrate saving and loading
#
# model = DDPG.load("ddpg_mountain")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
