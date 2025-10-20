# 🚕 Taxi-v3 with Dynamic Programming (DP)

**Policy Evaluation · Policy Iteration · Value Iteration** on Gymnasium's Taxi-v3.

This project computes optimal policies in a tabular MDP using exact model-based DP and then evaluates/visualizes performance with animations, plots, and summary tables.

## ✨ What you get

Exact DP solvers for Taxi-v3:

- **Policy Evaluation** (iterative)
- **Policy Iteration** (evaluation + greedy improvement)
- **Value Iteration** (Bellman optimality backups + greedy extraction)
- Transition model from the env (`env.unwrapped.P`) to do exact backups
- Testing harness with episode rollouts, metrics (avg return, std, success rate, avg steps) and live animation (matplotlib)
- **Visualizations**: comparison bar charts + value-function heatmap slice
- Results table printed to console and pickled to `taxi_dp_results.pkl` for your report

## 🗂 Project structure

```
.
├── TaxiFinal.py              # Main script with DP algorithms and plots
├── README.md                 # (this file)
└── taxi_dp_results.pkl       # (created after running; results dict for report)
```

## 🧰 Requirements

- **Python ≥ 3.9**

### Packages

```bash
pip install "gymnasium[toy_text]" numpy matplotlib pandas pygame
```

**Notes:**
- `gymnasium[toy_text]` ensures Taxi-v3 is available.
- `pygame` helps with rendering on some systems.
- If you're on Conda:

```bash
conda install -c conda-forge gymnasium pygame numpy matplotlib pandas
```

## ▶️ How to run

```bash
python TaxiFinal.py
```

**What you'll see:**

1. Console logs for Policy Iteration and Value Iteration convergence (Δ residuals / iteration counts).
2. Test episodes (rewards, steps, success) for each learned policy.
3. Plots: bar charts comparing algorithms, and a value-function heatmap.
4. A pickle file `taxi_dp_results.pkl` with all results for your report.

**Tip:** The first 3 test episodes are animated in a small matplotlib window (when `render=True` inside the script).

## 🧪 What does the script do?

1. **Creates the environment** and reads the exact transition model `P`  
   (`env.unwrapped.P` → probabilities, next state, reward, done) — needed for exact DP.

2. **Runs Policy Iteration:** alternating Policy Evaluation + greedy Policy Improvement until policy is stable.

3. **Runs Value Iteration:** Bellman optimality sweeps on *V* until Δ < θ, then greedy policy extraction.

4. **Evaluates** each learned policy over episodes; prints Avg Reward, Std, Success Rate, Avg Steps and iterations to converge.

5. **Plots** a 2×2 comparison dashboard (rewards, success rate, convergence iterations, steps) and a value heatmap slice (Passenger=Red, Destination=Green).

## ⚙️ Configuration & flags

The environment supports optional stochastic toggles:

- `is_raining=True` → movement noise: 0.8 to intended direction, 0.1 left, 0.1 right.
- `fickle_passenger=True` → once after first pickup and moving one step away from the source, passenger switches destination with 30% probability.

To try them for evaluation (or training if you re-extract the model), edit the env creation in the script, e.g.:

```python
# After solver = TaxiDynamicProgramming(...)
solver.env = gym.make('Taxi-v3', is_raining=True, fickle_passenger=True)
solver._build_transition_model()  # refresh P from this env (for exact DP)
```

DP remains correct iff the `P` you use in backups reflects those probabilities.  
MC/TD can also be compared under these flags (they learn from samples).

## 📊 Interpreting the outputs

### Convergence logs

- **Policy Iteration:** "Policy Evaluation converged in … iterations" multiple times, plus a final "Policy Iteration converged in N iterations!" (outer loops).
- **Value Iteration:** "… converged in M iterations!" once residual Δ < θ.

### Results table

Shows **Convergence** (iterations), **Avg Reward**, **Std Reward**, **Success Rate (%)**, **Avg Steps** per algorithm. Higher Avg Reward / lower Avg Steps is better on Taxi-v3.

### Plots

- **Rewards & Steps bars:** performance comparison.
- **Success Rate bar:** reliability.
- **Convergence iterations:** computational effort.
- **Heatmap:** value *V(s)* across taxi positions for a fixed passenger/destination slice (Passenger=Red, Destination=Green). Brighter = better start states.

## 🧩 Reproducibility tips

- Fix Python and Gymnasium versions in your environment (e.g., `requirements.txt`).
- If you need fixed episode behavior, add seeds in the code:

```python
state, _ = test_env.reset(seed=42)
np.random.seed(42)
```

- On headless Linux, you may need a non-interactive matplotlib backend:

```python
import matplotlib
matplotlib.use("Agg")
```

## 🛠 Troubleshooting

- **Plots don't show** → Ensure you run as a script (`python TaxiFinal.py`) or enable interactive backends.
- **Pygame warning on Windows** → It's benign; you can ignore it or upgrade setuptools/pygame.
- **Very slow first evaluation** → Normal for random initial policy; later evaluations converge faster.

---

**Happy reinforcement learning! 🎓🤖**
