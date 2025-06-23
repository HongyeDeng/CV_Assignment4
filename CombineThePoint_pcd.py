import open3d as o3d
import numpy as np
import os

# Define the directory containing the .pcd files to combine
input_parts_dir = r"PointClouds\B_segmented_regions" # e.g., where your segmented PCDs are saved
output_parts_dir = r"PointClouds\IntermediateFile_B" # e.g., where you want to save the combined PCD
# Define the path and name for the single output .pcd file
output_combined_file_path = os.path.join(output_parts_dir, "combined_segmented_cloud.pcd") # Saves in the same folder, or specify another

try:
    # Get a list of all .pcd files in the input directory
    pcd_files_to_combine = [f for f in os.listdir(input_parts_dir) if f.lower().endswith('.pcd')]

    if not pcd_files_to_combine:
        print(f"No .pcd files found in the '{input_parts_dir}' directory. Exiting.")
        exit()

    print(f"Found {len(pcd_files_to_combine)} .pcd files in '{input_parts_dir}'.")

    # Initialize an empty Open3D PointCloud object to store the combined points
    combined_pcd = o3d.geometry.PointCloud()
    total_points_combined = 0

    # Loop through each .pcd file, read its data, and add to the combined_pcd
    for i, filename in enumerate(pcd_files_to_combine):
        current_file_path = os.path.join(input_parts_dir, filename)
        print(f"--- Reading {i+1}/{len(pcd_files_to_combine)}: {filename} ---")
        
        try:
            current_pcd = o3d.io.read_point_cloud(current_file_path)
            num_points_current_file = len(current_pcd.points)
            print(f"  Loaded {num_points_current_file} points.")

            if num_points_current_file == 0:
                print(f"  File '{filename}' contains no points. Skipping.")
                continue
            
            # Add the current point cloud to the combined one
            # Open3D's PointCloud objects support direct addition for concatenation
            combined_pcd += current_pcd
            total_points_combined += num_points_current_file

        except Exception as e:
            print(f"  An error occurred while reading '{filename}': {e}. Skipping this file.")
            continue

    if total_points_combined == 0:
        print("No valid point clouds were read or combined. No output file will be created. Exiting.")
        exit()

    print(f"\n--- Combining complete. Total points in combined cloud: {total_points_combined} ---")

    # Save the combined PointCloud object to the new .pcd file
    print(f"Saving combined point cloud to: '{output_combined_file_path}'")
    o3d.io.write_point_cloud(output_combined_file_path, combined_pcd)
    print(f"Successfully combined points into '{output_combined_file_path}'")

except FileNotFoundError:
    print(f"Error: The input directory '{input_parts_dir}' not found.")
    print("Please ensure the directory exists and contains .pcd files.")
except ImportError:
    print("\n--- CRITICAL ERROR: Library Not Found ---")
    print("The 'open3d' library is not installed.")
    print("Please install it using: pip install open3d numpy")
    print("-----------------------------------------")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    print("Please check your input files, directory paths, and environment setup.")