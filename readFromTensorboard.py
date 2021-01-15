from tensorboard.backend.event_processing import event_accumulator
# cd logs\SAC_2_Ex3_EKF_gyro-v0_1

# 加载日志数据
ea = event_accumulator.EventAccumulator('logs\SAC_2_Ex3_EKF_gyro-v0_1\events.out.tfevents.1610628858.TUD259180')
ea.Reload()
print(ea.scalars.Keys())

episode_reward = ea.scalars.Items('episode_reward')
print(len(episode_reward))
print([(i.step, i.value) for i in episode_reward])

# Plot mean path of reference and state_of_interrest
if EVAL_PARAMS["merged"]:
    fig_1 = plt.figure(figsize=(9, 6), num=f"LAC_TF115_1")
    ax = fig_1.add_subplot(111)
    colors = "bgrcmk"
    cycol = cycle(colors)
for i in range(0, max(soi_mean_path.shape[0], ref_mean_path.shape[0])):
    if (i + 1) in req_ref or not req_ref:
        if not EVAL_PARAMS["merged"]:
            fig_1 = plt.figure(figsize=(9, 6), num=f"LAC_TF115_{i + 1}", )
            ax = fig_1.add_subplot(111)
            color1 = "red"
            color2 = "blue"
        else:
            color1 = next(cycol)
            color2 = color1
        t = range(max(eval_paths["episode_length"]))
        if i <= (len(soi_mean_path) - 1):
            ax.plot(
                t,
                soi_mean_path[i],
                color=color1,
                linestyle="dashed",
                label=f"state_of_interest_{i + 1}_mean",
            )
            if not EVAL_PARAMS["merged"]:
                ax.set_title(f"States of interest and reference {i + 1}")
            ax.fill_between(
                t,
                soi_mean_path[i] - soi_std_path[i],
                soi_mean_path[i] + soi_std_path[i],
                color=color1,
                alpha=0.3,
                label=f"state_of_interest_{i + 1}_std",
            )
        if i <= (len(ref_mean_path) - 1):
            ax.plot(
                t,
                ref_mean_path[i],
                color=color2,
                # label=f"reference_{i+1}",
            )
        if not EVAL_PARAMS["merged"]:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    if EVAL_PARAMS["merged"]:
        ax.set_title("True and Estimated Quatonian")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
