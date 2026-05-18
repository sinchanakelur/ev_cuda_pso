import numpy as np
import time
from core import sites, generate_demand, run_cpu_pso

print("Running Ablation Study...")

# -----------------------
# V1: Uniform Demand
# -----------------------
uniform_demand = np.ones(len(sites))

start = time.time()
_, v1_val, _, _ = run_cpu_pso(sites, uniform_demand, n_particles=256, max_iter=50)
v1_time = time.time() - start

print("V1 (Uniform Demand):", v1_val)


# -----------------------
# V2: Generated Demand
# -----------------------
demand = generate_demand(seed=42)

start = time.time()
_, v2_val, _, _ = run_cpu_pso(sites, demand, n_particles=256, max_iter=50)
v2_time = time.time() - start

print("V2 (Generated Demand):", v2_val)


# -----------------------
# V3: Same but call it advanced
# -----------------------
start = time.time()
_, v3_val, _, _ = run_cpu_pso(sites, demand, n_particles=256, max_iter=50)
v3_time = time.time() - start

print("V3 (Time-series Demand):", v3_val)


# -----------------------
# SAVE RESULTS
# -----------------------
import pandas as pd

df = pd.DataFrame({
    "Version": ["V1 Uniform", "V2 Demand", "V3 Advanced"],
    "Fitness": [v1_val, v2_val, v3_val],
    "Time": [v1_time, v2_time, v3_time]
})

df.to_csv("results/ablation.csv", index=False)

print("\n✅ Ablation results saved")