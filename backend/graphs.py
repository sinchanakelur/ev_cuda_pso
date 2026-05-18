import os
import pandas as pd

# Create results folder if not exists
os.makedirs("result", exist_ok=True)

# Save raw results
df = pd.DataFrame({
    "CPU_Fitness": cpu_results,
    "GPU_Fitness": gpu_results,
    "CPU_Time": cpu_times,
    "GPU_Time": gpu_times
})

df.to_csv("results/results.csv", index=False)

# Save summary
summary = pd.DataFrame({
    "Metric": ["CPU Mean", "GPU Mean", "CPU Time", "GPU Time", "Speedup"],
    "Value": [cpu_mean, gpu_mean, cpu_time_mean, gpu_time_mean, speedup]
})

summary.to_csv("result/summary.csv", index=False)

print("\n✅ Results saved in 'results/' folder")