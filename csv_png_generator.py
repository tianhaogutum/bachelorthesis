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

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_csv_files(folder_path):
    """
    Processes all CSV files in the specified folder, generating KDE plots, histograms,
    and vertical axis plots for the last column of each CSV file.

    Parameters:
    folder_path (str): The path to the folder containing the CSV files.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_path)
        last_column = df.iloc[:, -1]
        base_filename = os.path.splitext(csv_file)[0]
        
        plt.figure()
        sns.kdeplot(last_column, fill=True)
        plt.title(f'KDE Plot of {base_filename}')
        kde_output_path = os.path.join(folder_path, f'{base_filename}_kde.png')
        plt.savefig(kde_output_path)
        plt.close()
        
        plt.figure()
        plt.hist(last_column, bins=30, alpha=0.75)
        plt.title(f'Histogram of {base_filename}')
        hist_output_path = os.path.join(folder_path, f'{base_filename}_hist.png')
        plt.savefig(hist_output_path)
        plt.close()
        
        plt.figure(figsize=(4, 10))
        plt.scatter([0]*len(last_column), last_column, s=10, color='blue')
        plt.ylabel('Fitness Values')
        plt.title(f'Vertical Axis Plot of {base_filename}')
        plt.ylim(-100, 0)  
        plt.xticks([])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        vertical_output_path = os.path.join(folder_path, f'{base_filename}_vertical_axis.png')
        plt.savefig(vertical_output_path, dpi=300)
        plt.close()
        
        print(f'Processed {csv_file}: KDE, Histogram, and Vertical Axis plot saved.')

def main():
    """
    Reads the input file, extracts simulation variables and evaluation results,
    processes the data, and saves it to CSV files. Then, it calls process_csv_files
    to generate plots.
    """
    input_file = 'stat_results_collision_obst_quad_min.txt'
    output_folder = 'stat_results_analyse'

    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.read()

    pattern = r'simulation variables:\s*(\[\[.*?\]\])\s*evaluation:\s*({.*?)(?=simulation variables:|$)'
    simulations = re.findall(pattern, data, re.DOTALL)

    all_data = []

    for idx, (variables_str, evaluation_str) in enumerate(simulations, 1):
        variables_content = variables_str.replace('\n', ' ').replace('  ', ' ')
        variables_content = variables_content.strip('[]')
        variables_numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', variables_content)
        variables_array = np.array(variables_numbers, dtype=float)

        num_vars = 7
        num_rows = len(variables_array) // num_vars
        if len(variables_array) % num_vars != 0:
            print(f'Block {idx} variables data cannot be divided into {num_vars} columns')
            continue
        variables_array = variables_array.reshape((num_rows, num_vars))

        F_match = re.search(r"'F':\s*array\(\s*(\[\[.*?\]\])", evaluation_str, re.DOTALL)
        if F_match:
            F_str = F_match.group(1)
            F_content = F_str.replace('\n', ' ').replace('  ', ' ')
            F_content = F_content.strip('[]')
            F_numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', F_content)
            F_array = np.array(F_numbers, dtype=float)
        else:
            print(f'Block {idx} F values not found')
            continue

        if variables_array.shape[0] != F_array.shape[0]:
            print(f'Block {idx} variables and F values count do not match')
            continue

        data_combined = np.hstack((variables_array, F_array.reshape(-1, 1)))
        all_data.append(data_combined)

        columns = [f'Var{i+1}' for i in range(num_vars)] + ['F']

        df = pd.DataFrame(data_combined, columns=columns)
        csv_filename = os.path.join(output_folder, f'simulation_{idx}.csv')
        df.to_csv(csv_filename, index=False)
        print(f'Saved {csv_filename}')

    if all_data:
        all_data_combined = np.vstack(all_data)
        df_combined = pd.DataFrame(all_data_combined, columns=columns)
        combined_csv_filename = os.path.join(output_folder, 'combined.csv')
        df_combined.to_csv(combined_csv_filename, index=False)
        print(f'Saved combined CSV file: {combined_csv_filename}')
    else:
        print('No data to save')

if __name__ == '__main__':
    main()
    folder_path = './stat_results_analyse'
    process_csv_files(folder_path)