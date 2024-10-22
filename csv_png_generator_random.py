"""
MIT License

© 2024 Tianhao Gu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

input_file = 'stat_results_collision_obst_quad_min_random.txt'
output_dir = './stat_results_analyse_random'
os.makedirs(output_dir, exist_ok=True)

# Reading the input file
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to extract 'F' array
f_pattern = r"'F':\s*array\(\s*\[\[(.*?)\]\]\s*\)"
match = re.search(f_pattern, content, re.DOTALL)

if not match:
    raise ValueError("The 'F' array was not found in the file.")

# Extracting numerical values from the 'F' array
f_content = match.group(1)
numbers = re.findall(r'-?\d+\.?\d*', f_content)  # This will match both integers and floats
f_values = [float(num) for num in numbers]

if not f_values:
    raise ValueError("No numerical values were extracted from the 'F' array.")

# Track original indices and sort by values
f_values_with_indices = [(index + 1, value) for index, value in enumerate(f_values)]
f_values_sorted_with_indices = sorted(f_values_with_indices, key=lambda x: x[1])

# Save sorted f_values along with their original indices to CSV
df_with_index = pd.DataFrame(f_values_sorted_with_indices, columns=['Original Index', 'F Values'])

# Save the new DataFrame with the original index and sorted values
csv_output_with_index_path = os.path.join(output_dir, 'F_values_sorted_with_original_index.csv')
df_with_index.to_csv(csv_output_with_index_path, index=False)
print(f"CSV with sorted F values and original indices saved to {csv_output_with_index_path}")

# Extract sorted values for the plots
f_values_sorted = [x[1] for x in f_values_sorted_with_indices]

# Vertical axis plot
plt.figure(figsize=(4, 10))
plt.scatter([0]*len(f_values_sorted), f_values_sorted, s=10, color='blue')
plt.ylabel('Fitness Values')
plt.title('Vertical Axis Plot')
plt.xticks([])
plt.ylim(-100, 0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
vertical_output_path = os.path.join(output_dir, 'F_values_vertical_axis.png')
plt.savefig(vertical_output_path, dpi=300)
plt.close()

# KDE plot
plt.figure()
sns.kdeplot(f_values_sorted, fill=True)
plt.title('KDE Plot of F Values')
plt.xlabel('F Values')
plt.ylabel('Density')
kde_output_path = os.path.join(output_dir, 'F_values_kde.png')
plt.savefig(kde_output_path, dpi=300)
plt.close()

# Histogram plot
plt.figure()
plt.hist(f_values_sorted, bins=30, alpha=0.75, color='green')
plt.title('Histogram of F Values')
plt.xlabel('F Values')
plt.ylabel('Frequency')
hist_output_path = os.path.join(output_dir, 'F_values_histogram.png')
plt.savefig(hist_output_path, dpi=300)
plt.close()

print(f"Images saved to {output_dir}")
