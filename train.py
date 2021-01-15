import gym
import numpy as np
import matplotlib.pyplot as plt
import os

train = True #True #False
inference = False
model_num = 1
algorithm = "SAC"   # PPO2, SAC, DDPG, PPO1
env_name = 'Ex3_EKF_gyro-v0'  # 'Ex3_EKF_gyro-v0', 'Pendulum-v0','Ex3_pureEKF_gyro'
total_timesteps_ = 3000000  # 3000000 for sac,  500000 for ppo2, 1500000 for ddpg
exp_num = "2"
tensorboard_log_name = algorithm+"_"+exp_num+"_"+env_name
tensorboard_log_dir = "./logs/"
# tensorboard --logdir=PPO2_1_Ex3_EKF_gyro-v0_1 --port=6006 --host=127.0.0.1
# tensorboard --logdir=sac_ekf_3_3 --port=6007 --host=127.0.0.2
model_save_name = tensorboard_log_name+"_model_"+exp_num

if algorithm == "PPO2":
    from itertools import cycle
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common import make_vec_env
    from stable_baselines import PPO2
    env = make_vec_env(env_name, n_envs=3)
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log_dir)
    if train:
        for i in range(model_num):
            model.learn(total_timesteps=total_timesteps_, tb_log_name=tensorboard_log_name)
            model.save(model_save_name)
elif algorithm == "PPO1":
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines import PPO1
    env = gym.make(env_name)
    model = PPO1(MlpPolicy, env, verbose=1)
    if train:
        for i in range(model_num):
            model.learn(total_timesteps=total_timesteps_)
            model.save(model_save_name)
elif algorithm == "SAC":
    from stable_baselines.sac.policies import MlpPolicy
    from stable_baselines import SAC
    env = gym.make(env_name)
    model = SAC(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log_dir)
    if train:
        for i in range(model_num):
            model.learn(total_timesteps=total_timesteps_, log_interval=10, tb_log_name=tensorboard_log_name)
            model.save(model_save_name)
elif algorithm == "DDPG":
    if train:
        for i in range(model_num):
            from stable_baselines.ddpg.policies import MlpPolicy
            from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
            from stable_baselines import DDPG
            env = gym.make(env_name)

            # the noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            param_noise = None
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                        sigma=float(0.5) * np.ones(n_actions))

            model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise,
                         action_noise=action_noise, tensorboard_log=tensorboard_log_dir,
                         )
            model.learn(total_timesteps=total_timesteps_, log_interval=1, tb_log_name=tensorboard_log_name)
            model.save(model_save_name)

del model # remove to demonstrate saving and loading

if inference:
    if algorithm == "PPO2":
        env = make_vec_env('Ex3_EKF_gyro-v0')
        model = PPO2.load("ppo2_ekf_0")
    elif algorithm == "SAC":
        env = gym.make('Ex3_EKF_gyro-v0')
        model = SAC.load("sac_ekf_model_0")
    # Enjoy trained agent
    num_of_paths = 1
    max_ep_steps = 1000
    save_figs = False
    LOG_PATH = "./logs"
    fig_file_type = "pdf"
    roll_out_paths = {}
    roll_out_paths = {
        "s": [],
        "r": [],
        "s_": [],
        "state_of_interest": [],
        "reference": [],
        "episode_length": [],
        "return": [],
        "death_rate": 0.0,
    }
    for i in range(num_of_paths):

        # Path storage buckets
        episode_path = {
            "s": [],
            "r": [],
            "s_": [],
            "state_of_interest": [],
            "reference": [],
        }
        # while not dones[0]:
        s = env.reset()
        for j in range(max_ep_steps):
            action, _states = model.predict(s)
            s_, rewards, dones, infos = env.step(action)
            # Store observations
            episode_path["s"].append(s)
            episode_path["r"].append(rewards)
            episode_path["s_"].append(s_)
            info = infos[0]
            if "state_of_interest" in info.keys():
                episode_path["state_of_interest"].append(
                    np.array([info["state_of_interest"]])
                )
            if "reference" in info.keys():
                episode_path["reference"].append(np.array(info["reference"]))

            # Terminate if max step has been reached
            if j == (max_ep_steps-1):
                dones[0] = True
            s = s_

            # Check if episode is done and break loop
            if dones[0]:
                break

        # Append paths to paths list
        roll_out_paths["s"].append(episode_path["s"])
        roll_out_paths["r"].append(episode_path["r"])
        roll_out_paths["s_"].append(episode_path["s_"])
        roll_out_paths["state_of_interest"].append(
            episode_path["state_of_interest"]
        )
        roll_out_paths["reference"].append(episode_path["reference"])
        roll_out_paths["episode_length"].append(len(episode_path["s"]))
        roll_out_paths["return"].append(np.sum(episode_path["r"]))

    # Calculate roll_out death rate
    roll_out_paths["death_rate"] = sum(
        [
            episode <= (max_ep_steps - 1)
            for episode in roll_out_paths["episode_length"]
        ]) / len(roll_out_paths["episode_length"])

    mean_return = np.mean(roll_out_paths["return"])
    print('mean_return: ',mean_return)
    mean_episode_length = np.mean(
            roll_out_paths["episode_length"]
        )
    print('mean_episode_length: ',mean_episode_length)
    death_rate = roll_out_paths["death_rate"]
    print('death_rate: ',death_rate)

    print("Plotting states of reference...")
    print("Plotting mean path and standard deviation...")

    # Calculate mean path of reference and state_of_interest
    soi_trimmed = [
        path
        for path in roll_out_paths["state_of_interest"]
        if len(path) == max(roll_out_paths["episode_length"])
    ]  # Needed because unequal paths # FIXME: CLEANUP
    ref_trimmed = [
        path
        for path in roll_out_paths["reference"]
        if len(path) == max(roll_out_paths["episode_length"])
    ]  # Needed because unequal paths # FIXME: CLEANUP
    soi_mean_path = np.transpose(
        np.squeeze(np.mean(np.array(soi_trimmed), axis=0))
    )
    soi_std_path = np.transpose(
        np.squeeze(np.std(np.array(soi_trimmed), axis=0))
    )
    ref_mean_path = np.transpose(
        np.squeeze(np.mean(np.array(ref_trimmed), axis=0))
    )
    ref_std_path = np.transpose(
        np.squeeze(np.std(np.array(ref_trimmed), axis=0))
    )

    # Make sure arrays are right dimension
    soi_mean_path = (
        np.expand_dims(soi_mean_path, axis=0)
        if len(soi_mean_path.shape) == 1
        else soi_mean_path
    )
    soi_std_path = (
        np.expand_dims(soi_std_path, axis=0)
        if len(soi_std_path.shape) == 1
        else soi_std_path
    )
    ref_mean_path = (
        np.expand_dims(ref_mean_path, axis=0)
        if len(ref_mean_path.shape) == 1
        else ref_mean_path
    )
    ref_std_path = (
        np.expand_dims(ref_std_path, axis=0)
        if len(ref_std_path.shape) == 1
        else ref_std_path
    )


    # Plot mean path of reference and state_of_interest
    fig_1 = plt.figure(
        figsize=(9, 6), num=f"state-q-ppo2"
    )
    ax = fig_1.add_subplot(111)
    colors = "bgrcmk"
    cycol = cycle(colors)
    for i in range(0, min(soi_mean_path.shape[0], ref_mean_path.shape[0])):
        color1 = next(cycol)
        color2 = color1
        t = [i / 100.0 for i in range(0, max(roll_out_paths["episode_length"]))]
        if i <= (len(soi_mean_path) - 1):
            ax.plot(
                t,
                soi_mean_path[i],
                color=color1,
                linestyle="dashed",
                # label=f"state_of_interest_{i+1}_mean",
            )
            ax.fill_between(
                t,
                soi_mean_path[i] - soi_std_path[i],
                soi_mean_path[i] + soi_std_path[i],
                color=color1,
                alpha=0.3,
                # label=f"state_of_interest_{i+1}_std",
            )
        path = np.concatenate(
            [np.transpose(ref_mean_path), np.transpose(soi_mean_path), np.transpose(soi_std_path)], 1)
        # np.savetxt('inferenceResult-52.csv', path, delimiter=',')
        if i <= (len(ref_mean_path) - 1):
            ax.plot(
                t,
                ref_mean_path[i],
                color=color2,
                # label=f"reference_{i+1}",
            )
        if i <= (len(ref_mean_path) - 1):
            # ax.plot(
            #     t,
            #     ref_mean_path[i+4],
            #     color=color2,
            #     linestyle="dotted",
            #     # label=f"reference_{i+1}",
            # )
            plt.ylabel("Quaternion", fontsize=20)
            plt.xlabel("Time(s)", fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
            # ax.fill_between(
            #     t,
            #     ref_mean_path[i] - ref_std_path[i],
            #     ref_mean_path[i] + ref_std_path[i],
            #     color=color2,
            #     alpha=0.3,
            #     label=f"reference_{i+1}_std",
            # )  # FIXME: remove
        ax.set_rasterized(True)

    # Also plot mean and std of the observations
    print("Plotting observations...")
    print("Plotting mean path and standard deviation...")


    # Create figure
    fig_2 = plt.figure(
        figsize=(9, 6), num="observation-ppo2"
    )
    colors = "bgrcmk"
    cycol = cycle(colors)
    ax2 = fig_2.add_subplot(111)

    # Calculate mean observation path and std
    obs_trimmed = [
        path
        for path in roll_out_paths["s"]
        if len(path) == max(roll_out_paths["episode_length"])
    ]
    obs_mean_path = np.transpose(
        np.squeeze(np.mean(np.array(obs_trimmed), axis=0))
    )
    obs_std_path = np.transpose(
        np.squeeze(np.std(np.array(obs_trimmed), axis=0))
    )
    t = range(max(roll_out_paths["episode_length"]))

    # Plot state paths and std
    for i in range(0, obs_mean_path.shape[0]):
        color = next(cycol)
        ax2.plot(
            t,
            obs_mean_path[i],
            color=color,
            linestyle="dashed",
            label=(f"s_{i + 1}"),
        )
        ax2.fill_between(
            t,
            obs_mean_path[i] - obs_std_path[i],
            obs_mean_path[i] + obs_std_path[i],
            color=color,
            alpha=0.3,
            label=(f"s_{i + 1}_std"),
        )
    ax2.set_title("Observations")
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles2, labels2, loc=2, fancybox=False, shadow=False)

    # Plot mean cost and std
    # Create figure
    fig_3 = plt.figure(
        figsize=(9, 6), num="return-ppo2"
    )
    ax3 = fig_3.add_subplot(111)

    # Calculate mean observation path and std
    cost_trimmed = [
        path
        for path in roll_out_paths["r"]
        if len(path) == max(roll_out_paths["episode_length"])
    ]
    cost_mean_path = np.transpose(
        np.squeeze(np.mean(np.array(cost_trimmed), axis=0))
    )
    cost_std_path = np.transpose(
        np.squeeze(np.std(np.array(cost_trimmed), axis=0))
    )
    t = range(max(roll_out_paths["episode_length"]))

    # Plot state paths and std
    ax3.plot(
        t, cost_mean_path, color="g", linestyle="dashed", label=("mean cost"),
    )
    ax3.fill_between(
        t,
        cost_mean_path - cost_std_path,
        cost_mean_path + cost_std_path,
        color="g",
        alpha=0.3,
        label=("mean cost std"),
    )
    ax3.set_title("Mean cost")
    handles3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(handles3, labels3, loc=2, fancybox=False, shadow=False)

    # Show figures
    plt.show()

    # Save figures to pdf if requested
    if save_figs:
        fig_1.savefig(
            os.path.join(LOG_PATH, "Quatonian." + fig_file_type),
            bbox_inches="tight",
        )
        fig_2.savefig(
            os.path.join(LOG_PATH, "State." + fig_file_type),
            bbox_inches="tight",
        )
        fig_3.savefig(
            os.path.join(LOG_PATH, "Cost." + fig_file_type),
            bbox_inches="tight",
        )