import laspy
import numpy as np
import open3d as o3d
import os

# Define input and output directories
input_non_ground_dir = r"PointClouds\B_non_ground_parts" # Folder containing your non-ground LAS files
output_smooth_dir = r"PointClouds\B_smooth_non_ground_parts"         # Folder to save smooth points
output_rough_dir = r"PointClouds\B_rough_non_ground_parts"           # Folder to save rough points

# Create the output directories if they don't exist
os.makedirs(output_smooth_dir, exist_ok=True)
os.makedirs(output_rough_dir, exist_ok=True)
print(f"Ensuring output directory '{output_smooth_dir}' exists.")
print(f"Ensuring output directory '{output_rough_dir}' exists.")


# Parameters for roughness calculation and smoothness filtering
k_neighbors = 40  # Number of neighbors to consider for roughness calculation
smoothness_threshold = 0.008 # Points with roughness below or equal to this will be considered 'smooth'
                          # This value will likely require tuning based on your data and definition of 'smooth'.

try:
    # Get a list of all .las files in the input non-ground directory
    non_ground_files = [f for f in os.listdir(input_non_ground_dir) if f.lower().endswith('.las')]

    if not non_ground_files:
        print(f"No .las files found in the '{input_non_ground_dir}' directory. Please ensure files exist.")
    else:
        print(f"Found {len(non_ground_files)} non-ground .las files. Starting roughness calculation and separation...")

    for filename in non_ground_files:
        input_non_ground_file_path = os.path.join(input_non_ground_dir, filename)
        
        # Define output paths for smooth and rough files
        output_smooth_file_path = os.path.join(output_smooth_dir, f"smooth_{filename}")
        output_rough_file_path = os.path.join(output_rough_dir, f"rough_{filename}")

        print(f"\n--- Processing: {filename} ---")
        try:
            # 1. Read the non_ground LAS file
            las_data = laspy.read(input_non_ground_file_path)
            # num_points_in_file = las_data.points.count
            # print(f"  Read {num_points_in_file} non-ground points from '{filename}'.")

            # if num_points_in_file == 0:
            #     print(f"  No points found in '{filename}'. Skipping roughness calculation and saving.")
            #     continue

            # Get X, Y, Z coordinates
            xyz_points = np.vstack((las_data.X, las_data.Y, las_data.Z)).transpose()

            # Create an Open3D PointCloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_points)

            # Optional: If colors exist, copy them for initial visualization or later use
            if 'red' in las_data.point_format.dimension_names and \
               'green' in las_data.point_format.dimension_names and \
               'blue' in las_data.point_format.dimension_names:
                colors = np.vstack((las_data.red, las_data.green, las_data.blue)).transpose()
                if colors.max() > 255: colors = colors / 65535.0
                else: colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 2. Calculate roughness for each point
            print(f"  Calculating roughness for each point using {k_neighbors} neighbors...")
            
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            roughness_values = np.zeros(xyz_points.shape[0])

            for i in range(xyz_points.shape[0]):
                [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k_neighbors)
                neighbors = np.asarray(pcd.points)[idx, :]

                if neighbors.shape[0] >= 3: 
                    cov_matrix = np.cov(neighbors, rowvar=False)
                    eigenvalues = np.linalg.eigvalsh(cov_matrix)
                    eigenvalues = np.sort(eigenvalues)[::-1] # lambda_1 >= lambda_2 >= lambda_3

                    sum_eigenvalues = np.sum(eigenvalues)
                    if sum_eigenvalues > 1e-9:
                        roughness = eigenvalues[2] / sum_eigenvalues
                    else:
                        roughness = 0.0 
                    roughness_values[i] = roughness
                else:
                    roughness_values[i] = 0.0 # Assign 0 roughness if not enough neighbors

            print("  Roughness calculation complete.")

            # 3. Identify smooth and rough points
            smooth_indices = np.where(roughness_values <= smoothness_threshold)[0]
            rough_indices = np.where(roughness_values > smoothness_threshold)[0]

            print(f"  Identified {len(smooth_indices)} smooth points (roughness <= {smoothness_threshold}).")
            print(f"  Identified {len(rough_indices)} rough points (roughness > {smoothness_threshold}).")

            # 4. Save Smooth Points
            if len(smooth_indices) > 0:
                new_las_smooth = laspy.create(point_format=las_data.header.point_format,
                                              file_version=las_data.header.version)

                new_las_smooth.X = las_data.X[smooth_indices]
                new_las_smooth.Y = las_data.Y[smooth_indices]
                new_las_smooth.Z = las_data.Z[smooth_indices]

                for dim_name in las_data.point_format.dimension_names:
                    if dim_name.lower() not in ['x', 'y', 'z']:
                        setattr(new_las_smooth, dim_name, getattr(las_data, dim_name)[smooth_indices])

                new_las_smooth.header.x_offset = las_data.header.x_offset
                new_las_smooth.header.y_offset = las_data.header.y_offset
                new_las_smooth.header.z_offset = las_data.header.z_offset
                new_las_smooth.header.x_scale = las_data.header.x_scale
                new_las_smooth.header.y_scale = las_data.header.y_scale
                new_las_smooth.header.z_scale = las_data.header.z_scale

                new_las_smooth.header.x_min = np.min(new_las_smooth.X)
                new_las_smooth.header.x_max = np.max(new_las_smooth.X)
                new_las_smooth.header.y_min = np.min(new_las_smooth.Y)
                new_las_smooth.header.y_max = np.max(new_las_smooth.Y)
                new_las_smooth.header.z_min = np.min(new_las_smooth.Z)
                new_las_smooth.header.z_max = np.max(new_las_smooth.Z)

                print(f"  Saving smooth points to: '{output_smooth_file_path}'")
                new_las_smooth.write(output_smooth_file_path)
                print(f"  Successfully saved {len(smooth_indices)} smooth points from '{filename}'.")
            else:
                print(f"  No smooth points found for '{filename}'. No smooth file saved.")

            # 5. Save Rough Points
            if len(rough_indices) > 0:
                new_las_rough = laspy.create(point_format=las_data.header.point_format,
                                             file_version=las_data.header.version)

                new_las_rough.X = las_data.X[rough_indices]
                new_las_rough.Y = las_data.Y[rough_indices]
                new_las_rough.Z = las_data.Z[rough_indices]

                for dim_name in las_data.point_format.dimension_names:
                    if dim_name.lower() not in ['x', 'y', 'z']:
                        setattr(new_las_rough, dim_name, getattr(las_data, dim_name)[rough_indices])

                new_las_rough.header.x_offset = las_data.header.x_offset
                new_las_rough.header.y_offset = las_data.header.y_offset
                new_las_rough.header.z_offset = las_data.header.z_offset
                new_las_rough.header.x_scale = las_data.header.x_scale
                new_las_rough.header.y_scale = las_data.header.y_scale
                new_las_rough.header.z_scale = las_data.header.z_scale

                new_las_rough.header.x_min = np.min(new_las_rough.X)
                new_las_rough.header.x_max = np.max(new_las_rough.X)
                new_las_rough.header.y_min = np.min(new_las_rough.Y)
                new_las_rough.header.y_max = np.max(new_las_rough.Y)
                new_las_rough.header.z_min = np.min(new_las_rough.Z)
                new_las_rough.header.z_max = np.max(new_las_rough.Z)

                print(f"  Saving rough points to: '{output_rough_file_path}'")
                new_las_rough.write(output_rough_file_path)
                print(f"  Successfully saved {len(rough_indices)} rough points from '{filename}'.")
            else:
                print(f"  No rough points found for '{filename}'. No rough file saved.")

        except FileNotFoundError:
            print(f"  Error: Input file '{input_non_ground_file_path}' not found. Skipping.")
        except Exception as e:
            print(f"  An error occurred while processing '{filename}': {e}. Skipping.")

except ImportError:
    print("\n--- CRITICAL ERROR: Library Not Found ---")
    print("The 'laspy' or 'open3d' library is not installed.")
    print("Please install them using: pip install laspy numpy open3d")
    print("-----------------------------------------")
except Exception as e:
    print(f"\nAn unexpected error occurred during overall process: {e}")
    print("Please check your input directories and file permissions.")