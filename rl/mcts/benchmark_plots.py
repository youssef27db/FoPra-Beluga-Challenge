#!/usr/bin/env python3
"""
Benchmark Visualization Script for MCTS Python vs C++ Performance
Creates bar plots comparing optimized steps and calculation time.
"""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data from the comprehensive tests
benchmark_data = {
    'jig_sizes': [10, 20, 43, 61, 78, 103, 137],
    'python_time': [2.31, 5.29, 10.96, 37.88, 74.5, 238.6, 273.5],
    'cpp_time': [1.92, 3.37, 3.63, 9.00, 21.04, 68.6, 103.0],
    'python_opt_steps': [244, 486, 907, 1534, 2272, 7059, 7075],
    'cpp_opt_steps': [198, 561, 690, 1010, 2260, 6926, 7159],
    'python_speedup': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'cpp_speedup': [1.20, 1.57, 3.02, 4.21, 3.54, 3.48, 2.66]
}

# Set up the plotting style
plt.style.use('default')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('MCTS Performance Comparison: Python vs C++', fontsize=16, fontweight='bold')

# Colors for consistent styling
python_color = '#3498db'  # Blue
cpp_color = '#e74c3c'     # Red

# Bar width and positions
x = np.arange(len(benchmark_data['jig_sizes']))
width = 0.35

# Plot 1: Optimized Steps Comparison
bars1 = ax1.bar(x - width/2, benchmark_data['python_opt_steps'], width, 
                label='Python MCTS', color=python_color, alpha=0.8)
bars2 = ax1.bar(x + width/2, benchmark_data['cpp_opt_steps'], width,
                label='C++ MCTS', color=cpp_color, alpha=0.8)

ax1.set_xlabel('Problem Size (Jigs)')
ax1.set_ylabel('Optimized Steps')
ax1.set_title('Solution Quality: Optimized Steps by Implementation')
ax1.set_xticks(x)
ax1.set_xticklabels(benchmark_data['jig_sizes'])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7)

# Plot 2: Calculation Time Comparison
bars3 = ax2.bar(x - width/2, benchmark_data['python_time'], width,
                label='Python MCTS', color=python_color, alpha=0.8)
bars4 = ax2.bar(x + width/2, benchmark_data['cpp_time'], width,
                label='C++ MCTS', color=cpp_color, alpha=0.8)

ax2.set_xlabel('Problem Size (Jigs)')
ax2.set_ylabel('Calculation Time (seconds)')
ax2.set_title('Performance: Calculation Time by Implementation')
ax2.set_xticks(x)
ax2.set_xticklabels(benchmark_data['jig_sizes'])
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add value labels on bars for time
for i, bar in enumerate(bars3):
    height = bar.get_height()
    label = f'{height:.1f}s' if height < 60 else f'{height/60:.1f}m'
    ax2.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7, rotation=45 if height > 100 else 0)

for i, bar in enumerate(bars4):
    height = bar.get_height()
    label = f'{height:.1f}s' if height < 60 else f'{height/60:.1f}m'
    ax2.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7, rotation=45 if height > 100 else 0)

# Plot 3: Speedup Factor
bars5 = ax3.bar(x, benchmark_data['cpp_speedup'], width*1.5,
                label='C++ Speedup Factor', color='#27ae60', alpha=0.8)

ax3.set_xlabel('Problem Size (Jigs)')
ax3.set_ylabel('Speedup Factor (x times faster)')
ax3.set_title('C++ Performance Advantage: Speedup Factor vs Python')
ax3.set_xticks(x)
ax3.set_xticklabels(benchmark_data['jig_sizes'])
ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No speedup')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add speedup value labels
for i, bar in enumerate(bars5):
    height = bar.get_height()
    ax3.annotate(f'{height:.2f}x',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 4: Scaling Analysis (Linear scale for time)
ax4.plot(benchmark_data['jig_sizes'], benchmark_data['python_time'], 
         'o-', label='Python MCTS', color=python_color, linewidth=2, markersize=8)
ax4.plot(benchmark_data['jig_sizes'], benchmark_data['cpp_time'], 
         's-', label='C++ MCTS', color=cpp_color, linewidth=2, markersize=8)

ax4.set_xlabel('Problem Size (Jigs)')
ax4.set_ylabel('Calculation Time (seconds)')
ax4.set_title('Scalability Analysis: Time vs Problem Size')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add annotations for key data points
for i, (jigs, py_time, cpp_time) in enumerate(zip(benchmark_data['jig_sizes'], 
                                                 benchmark_data['python_time'], 
                                                 benchmark_data['cpp_time'])):
    if i % 2 == 0:  # Annotate every other point to avoid clutter
        ax4.annotate(f'{py_time:.1f}s', (jigs, py_time), xytext=(5, 10), 
                    textcoords='offset points', fontsize=8, color=python_color)
        ax4.annotate(f'{cpp_time:.1f}s', (jigs, cpp_time), xytext=(5, -15), 
                    textcoords='offset points', fontsize=8, color=cpp_color)

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add more space around plots

# Save the plots
output_path = 'mcts_benchmark_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print(f"✅ Benchmark plots saved to: {output_path}")

# Display summary statistics
print("\n📊 BENCHMARK SUMMARY:")
print(f"Problems tested: {len(benchmark_data['jig_sizes'])} (from {min(benchmark_data['jig_sizes'])} to {max(benchmark_data['jig_sizes'])} jigs)")
print(f"Average C++ speedup: {np.mean(benchmark_data['cpp_speedup']):.2f}x")
print(f"Maximum C++ speedup: {max(benchmark_data['cpp_speedup']):.2f}x (at {benchmark_data['jig_sizes'][benchmark_data['cpp_speedup'].index(max(benchmark_data['cpp_speedup']))]} jigs)")
print(f"Total time saved (largest problem): {benchmark_data['python_time'][-1] - benchmark_data['cpp_time'][-1]:.1f} seconds")

# Show the plots
plt.show()

if __name__ == "__main__":
    print("🚀 MCTS Benchmark Visualization")
    print("Generating comprehensive performance comparison plots...")
