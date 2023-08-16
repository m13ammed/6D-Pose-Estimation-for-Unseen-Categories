import re
import os

def read_inlier_ratio_from_file(file_path):
    with open(file_path, 'r') as txt_file:
        lines = txt_file.readlines()
        if len(lines) >= 2:
            # Extract numerical value using regular expression
            inlier_ratio_line = lines[1].strip()
            match = re.search(r'\d+\.\d+', inlier_ratio_line)
            if match:
                inlier_ratio = float(match.group())
                return inlier_ratio
    return None

def calculate_average_inlier_ratio(file_paths):
    inlier_ratios = []
    for file_path in file_paths:
        inlier_ratio = read_inlier_ratio_from_file(file_path)
        if inlier_ratio is not None:
            inlier_ratios.append(inlier_ratio)

    if inlier_ratios:
        average_inlier_ratio = sum(inlier_ratios) / len(inlier_ratios)
        return average_inlier_ratio
    return None


directory_path = "past_results/results_current_best_pbr_new/results_poses_RANSAC/results"  # Replace this with the actual directory path
file_paths = []

for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_paths.append(os.path.join(directory_path, filename))

#file_paths = ['past_results/results_125_pt_files/results_poses_RANSAC/results/obj_1_result_3.txt', 'past_results/results_125_pt_files/results_poses_RANSAC/results/obj_1_result_14.txt', 'past_results/results_125_pt_files/results_poses_RANSAC/results/obj_1_result_24.txt']

average_inlier_ratio = calculate_average_inlier_ratio(file_paths)
if average_inlier_ratio is not None:
    print("Average Inlier Ratio:", average_inlier_ratio)
else:
    print("No inlier ratios found in the files.")