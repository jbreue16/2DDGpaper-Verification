"""
Created November 2024

This script is used to convert the convergence data from json files to csv,
which can be easily processed by e.g. latex to generate tables.

@author: jmbr
"""

import json
import csv

def json_to_csv(json_file, csv_file, subgroup_path, ignore_data):
    # Read the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Navigate to the specified subgroup path
    subgroup = data
    for key in subgroup_path:
        if key not in subgroup:
            raise KeyError(f"Key '{key}' not found in JSON data at path {' -> '.join(subgroup_path[:subgroup_path.index(key)+1])}.")
        subgroup = subgroup[key]

    # Check if the subgroup is a dictionary with lists as values
    if not isinstance(subgroup, dict) or not all(isinstance(value, list) for value in subgroup.values()):
        raise ValueError(f"Subgroup at path '{' -> '.join(subgroup_path)}' must be a dictionary with lists as values.")

    # Extract keys and corresponding lists
    all_keys = list(subgroup.keys())
    all_keys = [item for item in all_keys if item not in ignore_data]

    # Prepare rows for the CSV
    rows = []
    max_length = max(len(values) for values in subgroup.values())
    for i in range(max_length):
        row = []
        for key in all_keys:
            row.append(subgroup[key][i] if i < len(subgroup[key]) else "")  # Fill missing values with an empty string
        rows.append(row)

    # Write to the CSV file
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(all_keys)

        # Write data rows
        writer.writerows(rows)

# Example usage
file_path = r"C:\Users\jmbr\software\2DGRM-Verification\output\test_cadet-core\2D_chromatography/"
json_file = file_path + r"convergence_2DGRM3Zone_noBnd_1Comp.json"  # Input JSON file
csv_file = file_path + r"convergence_2DGRM3Zone_noBnd_1Comp.csv"  # Output CSV file
subgroup_path = ['convergence', 'FV', 'outlet']  # Path to the subgroup in the JSON file
# ignore_data not required since desired columns can be picked in latex
ignore_data = []#['$N_d$', 'Min. value', 'DoF', 'Bulk DoF']

json_to_csv(json_file, csv_file, subgroup_path, ignore_data)

json_file = file_path + r"convergence_2DGRM3Zone_dynLin_1Comp.json"  # Input JSON file
csv_file = file_path + r"convergence_2DGRM3Zone_dynLin_1Comp.csv"  # Output CSV file
json_to_csv(json_file, csv_file, subgroup_path, ignore_data)

json_file = file_path + r"convergence_2DGRMsd3Zone_dynLin_1Comp.json"  # Input JSON file
csv_file = file_path + r"convergence_2DGRMsd3Zone_dynLin_1Comp.csv"  # Output CSV file
json_to_csv(json_file, csv_file, subgroup_path, ignore_data)

json_file = file_path + r"convergence_2DGRM3Zone_reqLin_1Comp.json"  # Input JSON file
csv_file = file_path + r"convergence_2DGRM3Zone_reqLin_1Comp.csv"  # Output CSV file
json_to_csv(json_file, csv_file, subgroup_path, ignore_data)

json_file = file_path + r"convergence_2DGRMsd3Zone_reqLin_1Comp.json"  # Input JSON file
csv_file = file_path + r"convergence_2DGRMsd3Zone_reqLin_1Comp.csv"  # Output CSV file
json_to_csv(json_file, csv_file, subgroup_path, ignore_data)

json_file = file_path + r"convergence_2DGRM2parType3Zone_1Comp.json"  # Input JSON file
csv_file = file_path + r"convergence_2DGRM2parType3Zone_1Comp.csv"  # Output CSV file
json_to_csv(json_file, csv_file, subgroup_path, ignore_data)

