import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import pandas as pd

# opcional: si quieres semillas reproducibles, descomenta:
# import random
# def set_seed(seed=0):
#     np.random.seed(seed); random.seed(seed)

class TaxiDynamicProgramming:
    """
    Dynamic Programming algorithms for solving the Taxi-v3 environment.

    The Taxi problem involves navigating a 5x5 grid to pick up and drop off
    passengers at designated locations (R, G, Y, B).
    """

    def __init__(self, env_name='Taxi-v3', gamma=0.99, theta=1e-6):
        """
        Initialize the DP solver for Taxi environment.

        Args:
            env_name (str): Name of the Gymnasium environment
            gamma (float): Discount factor for future rewards
            theta (float): Convergence threshold for iterative algorithms
        """
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.theta = theta

        # Environment properties
        self.n_states = self.env.observation_space.n  # 500 states
        self.n_actions = self.env.action_space.n      # 6 actions

        # Initialize value function and policy
        self.V = np.zeros(self.n_states)
        self.policy = np.ones([self.n_states, self.n_actions]) / self.n_actions

        # Build transition model P[s,a] -> list of (prob, next_state, reward, done)
        self._build_transition_model()

        # For visualization
        self.episode_rewards = []
        self.convergence_history = []   # will store delta per sweep (for plots)

    def _build_transition_model(self):
        """
        Build the transition probability model using the environment's P attribute.
        Gymnasium provides this directly for Taxi-v3.
        """
        print("Building transition model for Taxi environment...")
        self.P = self.env.unwrapped.P
        print(f"Transition model built: {self.n_states} states, {self.n_actions} actions")

    def policy_evaluation(self, policy=None, max_iterations=1000):
        """
        Evaluate a given policy using iterative policy evaluation.

        Args:
            policy: Policy to evaluate (if None, uses current policy)
            max_iterations: Maximum number of iterations

        Returns:
            V: State value function
        """
        if policy is None:
            policy = self.policy

        V = np.zeros(self.n_states)
        iteration = 0

        print("\nStarting Policy Evaluation...")

        self.convergence_history = []
        while iteration < max_iterations:
            delta = 0
            V_old = V.copy()

            for s in range(self.n_states):
                v = 0.0
                for a in range(self.n_actions):
                    for prob, next_state, reward, done in self.P[s][a]:
                        v += policy[s, a] * prob * (reward + self.gamma * V_old[next_state] * (1 - done))
                V[s] = v
                delta = max(delta, abs(V_old[s] - V[s]))

            iteration += 1
            self.convergence_history.append(delta)

            if delta < self.theta:
                print(f"Policy Evaluation converged in {iteration} iterations")
                break

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Delta: {delta:.6f}")

        return V

    def policy_improvement(self, V):
        """
        Improve policy based on current value function.

        Args:
            V: Current state value function

        Returns:
            new_policy: Improved policy
            policy_stable: Whether policy has changed
        """
        new_policy = np.zeros([self.n_states, self.n_actions])

        for s in range(self.n_states):
            # Compute action values for current state
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for prob, next_state, reward, done in self.P[s][a]:
                    action_values[a] += prob * (reward + self.gamma * V[next_state] * (1 - done))
            best_action = np.argmax(action_values)
            new_policy[s, best_action] = 1.0

        policy_stable = np.array_equal(new_policy, self.policy)
        return new_policy, policy_stable

    def policy_iteration(self, max_iterations=100, return_history=False):
        """
        Solve the environment using Policy Iteration algorithm.

        Returns:
            policy: Optimal policy
            V: Optimal value function
            iterations: Number of outer iterations until convergence
            (optional) pi_history: concatenated deltas from policy evaluation sweeps
        """
        print("\n" + "="*60)
        print("POLICY ITERATION")
        print("="*60)

        self.policy = np.ones([self.n_states, self.n_actions]) / self.n_actions
        iterations = 0
        pi_history = []  # concatenated deltas of policy evaluation (for convergence plot)

        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            V = self.policy_evaluation(self.policy)     # fills self.convergence_history
            pi_history.extend(self.convergence_history)  # concatenate evaluation deltas

            new_policy, policy_stable = self.policy_improvement(V)
            iterations += 1

            if policy_stable:
                print(f"\n✓ Policy Iteration converged in {iterations} iterations!")
                self.policy = new_policy
                self.V = V
                if return_history:
                    return self.policy, V, iterations, pi_history
                return self.policy, V, iterations

            self.policy = new_policy

        print(f"\nPolicy Iteration completed {max_iterations} iterations")
        self.V = V
        if return_history:
            return self.policy, V, iterations, pi_history
        return self.policy, V, iterations

    def value_iteration(self, max_iterations=1000, return_history=False):
        """
        Solve the environment using Value Iteration algorithm.

        Returns:
            policy: Optimal policy
            V: Optimal value function
            iterations: Number of iterations until convergence
            (optional) vi_history: list of deltas per sweep
        """
        print("\n" + "="*60)
        print("VALUE ITERATION")
        print("="*60)

        V = np.zeros(self.n_states)
        iterations = 0
        self.convergence_history = []

        for _ in range(max_iterations):
            delta = 0.0
            V_old = V.copy()

            for s in range(self.n_states):
                action_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for prob, next_state, reward, done in self.P[s][a]:
                        action_values[a] += prob * (reward + self.gamma * V_old[next_state] * (1 - done))
                best_action_value = np.max(action_values)
                delta = max(delta, abs(best_action_value - V[s]))
                V[s] = best_action_value

            iterations += 1
            self.convergence_history.append(delta)

            if delta < self.theta:
                print(f"\n✓ Value Iteration converged in {iterations} iterations!")
                break

            if iterations % 100 == 0:
                print(f"Iteration {iterations}, Delta: {delta:.6f}")

        # Extract optimal policy from value function
        policy = np.zeros([self.n_states, self.n_actions])
        for s in range(self.n_states):
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for prob, next_state, reward, done in self.P[s][a]:
                    action_values[a] += prob * (reward + self.gamma * V[next_state] * (1 - done))
            best_action = np.argmax(action_values)
            policy[s, best_action] = 1.0

        self.V = V
        self.policy = policy
        if return_history:
            return policy, V, iterations, list(self.convergence_history)
        return policy, V, iterations

    def test_policy(self, policy, num_episodes=10, render=False, max_steps=200, record=False, gif_name="gifs/taxi_episode.gif"):
        """
        Test the learned policy and visualize the taxi behavior.

        Args:
            policy: Policy to test
            num_episodes: Number of test episodes
            render: Whether to render the environment
            max_steps: Maximum steps per episode
            record: If True and render=True, saves a GIF for the first episode
            gif_name: Path to save the GIF

        Returns:
            results: Dictionary with performance metrics
        """
        print("\n" + "="*60)
        print("TESTING LEARNED POLICY")
        print("="*60)

        # Create a separate environment for testing with render mode if needed
        if render:
            test_env = gym.make('Taxi-v3', render_mode='rgb_array')
            plt.ion()
            fig, ax = plt.subplots(figsize=(6, 6))
            img_display = None
            frames = []
        else:
            test_env = gym.make('Taxi-v3')

        rewards, successes, steps_list = [], [], []

        for episode in range(num_episodes):
            state, _ = test_env.reset()
            total_reward = 0
            steps = 0
            done = False
            success_flag = False

            print(f"\n--- Episode {episode + 1} ---")

            while not done and steps < max_steps:
                if render and episode < 3:
                    frame = test_env.render()
                    if img_display is None:
                        img_display = ax.imshow(frame)
                        ax.axis('off')
                    else:
                        img_display.set_data(frame)
                    ax.set_title(f'Episode: {episode + 1}/{num_episodes} | Step: {steps + 1} | Reward: {total_reward}',
                                 fontsize=10, fontweight='bold')
                    plt.pause(0.25)
                    if record and episode == 0:
                        frames.append(frame)

                action = np.argmax(policy[state])
                next_state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                if terminated:
                    success_flag = True
                total_reward += reward
                steps += 1
                state = next_state

            rewards.append(total_reward)
            successes.append(1 if success_flag else 0)
            steps_list.append(steps)
            print(f"Episode {episode + 1} finished: Reward = {total_reward}, Steps = {steps}, Success = {success_flag}")

        if render:
            plt.ioff()
            plt.close(fig)
            if record and frames:
                os.makedirs(os.path.dirname(gif_name), exist_ok=True)
                try:
                    import imageio.v2 as imageio
                    imageio.mimsave(gif_name, frames, fps=3)
                    print(f"GIF saved to {gif_name}")
                except Exception as e:
                    print(f"(Optional) Could not save GIF: {e}")

        test_env.close()

        results = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': np.mean(successes) * 100,
            'avg_steps': np.mean(steps_list),
            'all_rewards': rewards
        }
        return results

    def plot_results(self, results_dict, save_path="figs/fig_results_dp_taxi.png"):
        """
        Plot comparison of different algorithms (bars) and save.

        Args:
            results_dict: Dictionary mapping algorithm names to results
            save_path: path to save the figure
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Average Rewards Comparison
        algorithms = list(results_dict.keys())
        avg_rewards = [results_dict[alg]['metrics']['avg_reward'] for alg in algorithms]
        std_rewards = [results_dict[alg]['metrics']['std_reward'] for alg in algorithms]

        axes[0, 0].bar(algorithms, avg_rewards, yerr=std_rewards, capsize=5, alpha=0.7, color=['#2ecc71', '#3498db'])
        axes[0, 0].set_title('Average Reward per Episode', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Plot 2: Success Rate
        success_rates = [results_dict[alg]['metrics']['success_rate'] for alg in algorithms]
        axes[0, 1].bar(algorithms, success_rates, color=['#27ae60', '#2980b9'], alpha=0.7)
        axes[0, 1].set_title('Success Rate (%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim([0, 105])
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Plot 3: Convergence Speed (outer iterations)
        convergence_iters = [results_dict[alg]['iterations'] for alg in algorithms]
        axes[1, 0].bar(algorithms, convergence_iters, color=['#e67e22', '#d35400'], alpha=0.7)
        axes[1, 0].set_title('Convergence Speed (Iterations)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Iterations')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Plot 4: Average Steps per Episode
        avg_steps = [results_dict[alg]['metrics']['avg_steps'] for alg in algorithms]
        axes[1, 1].bar(algorithms, avg_steps, color=['#9b59b6', '#8e44ad'], alpha=0.7)
        axes[1, 1].set_title('Average Steps per Episode', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.suptitle('Dynamic Programming Algorithms Comparison - Taxi-v3', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        return fig

    def visualize_value_function(self, passenger_loc=0, destination=1, save_path="figs/fig_value_heatmap_taxi.png"):
        """
        Visualize the value function as a heatmap for a fixed passenger/destination.

        Args:
            passenger_loc: 0=Red, 1=Green, 2=Yellow, 3=Blue
            destination:   0=Red, 1=Green, 2=Yellow, 3=Blue
            save_path: path to save the figure
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        grid = np.zeros((5, 5))

        for row in range(5):
            for col in range(5):
                # Taxi state encoding: ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
                state = ((row * 5 + col) * 5 + passenger_loc) * 4 + destination
                if state < self.n_states:
                    grid[row, col] = self.V[state]

        plt.figure(figsize=(8, 6))
        im = plt.imshow(grid, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(im, label='State Value')
        plt.title('Value Function Heatmap\n(Passenger at Red, Destination: Green)', fontweight='bold')
        plt.xlabel('Column'); plt.ylabel('Row')

        # Grid lines
        for i in range(6):
            plt.axhline(y=i-0.5, color='black', linewidth=0.5)
            plt.axvline(x=i-0.5, color='black', linewidth=0.5)

        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()

    def visualize_policy_arrows(self, passenger_loc=0, destination=1, save_path="figs/policy_quiver.png"):
        """
        Quiver plot with greedy policy arrows on 5x5 grid (NSEW actions only are drawn).

        Args:
            passenger_loc: fixed passenger location for the slice
            destination:   fixed destination
            save_path:     path to save the figure
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        A = np.full((5, 5), -1)

        for r in range(5):
            for c in range(5):
                s = ((r * 5 + c) * 5 + passenger_loc) * 4 + destination
                if s < self.n_states:
                    A[r, c] = np.argmax(self.policy[s])

        # directions: 0=south, 1=north, 2=east, 3=west, 4/5 pickup/drop not drawn
        dr = {0: +1, 1: -1, 2: 0, 3: 0}
        dc = {0: 0,  1: 0,  2: +1, 3: -1}

        Y, X, U, V = [], [], [], []
        for r in range(5):
            for c in range(5):
                a = A[r, c]
                if a in (0, 1, 2, 3):
                    Y.append(r); X.append(c)
                    V.append(dr[a]); U.append(dc[a])

        plt.figure(figsize=(6, 5))
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
        plt.gca().invert_yaxis()
        plt.title('Greedy Policy Arrows (Passenger=R, Destination=G)')
        plt.xticks(range(5)); plt.yticks(range(5))
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()


def main():
    """
    Main execution function for Dynamic Programming algorithms on Taxi-v3.
    """
    print("="*60)
    print("TAXI-V3 DYNAMIC PROGRAMMING IMPLEMENTATION")
    print("="*60)
    print("\nInitializing environment and DP solver...")

    os.makedirs("figs", exist_ok=True)

    # Initialize solver
    solver = TaxiDynamicProgramming(gamma=0.99, theta=1e-6)

    # Dictionary to store results
    all_results = {}

    # 1) POLICY ITERATION
    print("\n" + "="*60)
    print("Running Policy Iteration...")
    print("="*60)
    policy_pi, V_pi, iters_pi, pi_deltas = solver.policy_iteration(return_history=True)
    results_pi = solver.test_policy(policy_pi, num_episodes=10, render=True, record=False)
    all_results['Policy Iteration'] = {
        'policy': policy_pi,
        'value': V_pi,
        'iterations': iters_pi,
        'metrics': results_pi
    }

    # 2) VALUE ITERATION
    print("\n" + "="*60)
    print("Running Value Iteration...")
    print("="*60)
    policy_vi, V_vi, iters_vi, vi_deltas = solver.value_iteration(return_history=True)
    results_vi = solver.test_policy(policy_vi, num_episodes=10, render=True, record=False)
    all_results['Value Iteration'] = {
        'policy': policy_vi,
        'value': V_vi,
        'iterations': iters_vi,
        'metrics': results_vi
    }

    # 3) SUMMARY TABLE
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    summary_data = []
    for alg_name, alg_results in all_results.items():
        summary_data.append({
            'Algorithm': alg_name,
            'Convergence (iterations)': alg_results['iterations'],
            'Avg Reward': f"{alg_results['metrics']['avg_reward']:.2f}",
            'Std Reward': f"{alg_results['metrics']['std_reward']:.2f}",
            'Success Rate (%)': f"{alg_results['metrics']['success_rate']:.1f}",
            'Avg Steps': f"{alg_results['metrics']['avg_steps']:.1f}"
        })
    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))

    # 4) BAR FIGURES (and save)
    solver.plot_results(all_results, save_path="figs/fig_results_dp_taxi.png")

    # 5) CONVERGENCE CURVES (and save)
    plt.figure()
    plt.semilogy(vi_deltas)
    plt.xlabel('Iteration')
    plt.ylabel('Max-norm residual Δ')
    plt.title('Value Iteration Convergence (Taxi-v3)')
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('figs/vi_convergence.png', dpi=200, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.semilogy(pi_deltas)
    plt.xlabel('Policy Evaluation sweeps (concatenated)')
    plt.ylabel('Max-norm residual Δ')
    plt.title('Policy Iteration – Evaluation Convergence (Taxi-v3)')
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('figs/pi_eval_convergence.png', dpi=200, bbox_inches='tight')
    plt.show()

    # 6) HEATMAP OF V (and save)
    solver.visualize_value_function(passenger_loc=0, destination=1, save_path="figs/fig_value_heatmap_taxi.png")

    # 7) POLICY ARROWS (and save)
    solver.visualize_policy_arrows(passenger_loc=0, destination=1, save_path="figs/policy_quiver.png")

    print("\n" + "="*60)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)

    return solver, all_results


if __name__ == "__main__":
    # set_seed(0)   # opcional
    solver, results = main()

    # Save results for report
    print("\nSaving results to file...")
    import pickle
    with open('taxi_dp_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to 'taxi_dp_results.pkl'")
