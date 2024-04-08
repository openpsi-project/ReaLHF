import matplotlib.pyplot as plt

# Example data
num_gpus = [8, 16, 32, 64]
small_model_performance = [100, 150, 200, 250]  # Example throughput values
medium_model_performance = [80, 140, 180, 220]  # Example throughput values
large_model_performance = [60, 120, 160, 200]  # Example throughput values

# Plotting
plt.figure(figsize=(10, 6))

# Plot for small model
plt.plot(num_gpus, small_model_performance, label='Small Model')

# Plot for medium model
plt.plot(num_gpus[1:], medium_model_performance[1:], label='Medium Model')

# Plot for large model
plt.plot(num_gpus[2:], large_model_performance[2:], label='Large Model')

plt.title('Scaling Trend of Different Applications')
plt.xlabel('Number of GPUs')
plt.ylabel('Performance Metric')
plt.xticks(num_gpus)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vss.png")