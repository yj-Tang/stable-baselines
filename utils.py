
import numpy as np

PARAMS = {
    "num_of_evaluation_paths" : 10
    "max_ep_steps": 800
    "eval_render": False
}


def training_evaluation(self):
    """Evaluates the performance of the current policy in
    several test rollouts.

    Returns:
        [type]: [description]
    """

    # Training setting
    total_cost_1 = []
    total_cost_2 = []
    episode_length = []
    die_count = 0
    seed_average_cost_1 = []
    seed_average_cost_2 = []

    # Perform roolouts to evaluate performance
    for i in range(PARAMS["num_of_evaluation_paths"]):
        cost_1 = 0
        cost_2 = 0
        if self.env.__class__.__name__.lower() == "ex3_ekf_gyro":
            s = self.env.reset(eval=True)
        else:
            s = self.env.reset()
        for j in range(PARAMS["max_ep_steps"]):
            if PARAMS["eval_render"]:
                self.env.render()


            action = self.policy_tf.step(s, deterministic=False).flatten()
            # Add noise to the action (improve exploration,
            # not needed in general)
            if self.action_noise is not None:
                action = np.clip(action + self.action_noise(), -1, 1)
            # inferred actions need to be transformed to environment action_space before stepping
            unscaled_action = unscale_action(self.action_space, action)

            assert action.shape == self.env.action_space.shape

            s_, r, done, _ = self.env.step(unscaled_action)
            cost_1 += r[0]
            cost_2 += r[1]
            if j == PARAMS["max_ep_steps"] - 1:
                done = True
            s = s_
            if done:
                seed_average_cost_1.append(cost_1)
                seed_average_cost_2.append(cost_2)
                episode_length.append(j)
                if j < PARAMS["max_ep_steps"] - 1:
                    die_count += 1
                break

    # Save evaluation results
    total_cost_1.append(np.mean(seed_average_cost_1))
    total_cost_2.append(np.mean(seed_average_cost_2))
    total_cost_mean_1 = np.average(total_cost_1)
    total_cost_mean_2 = np.average(total_cost_2)
    average_length = np.average(episode_length)

    # Return evaluation results
    diagnostic = {
        "test_return_1": total_cost_mean_1,
        "test_return_2": total_cost_mean_2,
        "test_average_length": average_length,
    }
    return diagnostic
