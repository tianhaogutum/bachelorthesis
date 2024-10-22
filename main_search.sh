#!/bin/bash

# Function to run Python script and check for errors
run_script() {
    echo "Running $1..."
    if python "$1"; then
        echo "$1 completed successfully."
    else
        echo "Error occurred while running $1."
        exit 1
    fi
}

# List of scripts to run in order
scripts=(
    "search_average_distance_with_obsts.py"
    "csv_png_generator.py"
    "search_average_distance_with_obsts_random.py"
    "csv_png_generator_random.py"
)

# Loop through and execute each script
for script in "${scripts[@]}"; do
    run_script "$script"
done

# Get the current date and time in the format YYYYMMDD_HHMMSS
timestamp=$(date +"%Y%m%d_%H%M%S")

# Create a 'results' directory with a timestamp suffix
results_dir="results_$timestamp"
echo "Creating results directory structure: $results_dir..."
mkdir -p "$results_dir/metaheuristics" "$results_dir/random"

# Move specific folders to the appropriate subdirectories
echo "Moving files to metaheuristics and random directories..."
mv pathOfDrones_average_distance_with_obsts stat_results_analyse stat_results_vis_average_distance_with_obsts "$results_dir/metaheuristics/"
mv pathOfDrones_random stat_results_analyse_random stat_results_vis_average_distance_with_obsts_random "$results_dir/random/"

# Delete specific text files
echo "Deleting unnecessary files..."
rm -f stat_results_average_distance_with_obsts.txt stat_results_random.txt

echo "All operations completed successfully. Results are saved in $results_dir."
