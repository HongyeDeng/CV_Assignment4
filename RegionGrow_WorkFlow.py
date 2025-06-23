import pclpy
from pclpy import pcl
import laspy
import numpy as np
import os

# --- Configuration ---
input_las_dir = r"PointClouds\B_smooth_non_ground_parts"  # Folder containing your .las files
output_segmented_dir = r"PointClouds\B_segmented_regions" # Folder to save the colored segmented .pcd files

# Ensure the output directory exists
os.makedirs(output_segmented_dir, exist_ok=True)
print(f"Ensuring output directory '{output_segmented_dir}' exists.")

# --- Region Growing Parameters (as provided by you) ---
# Normal Estimation parameters
normal_estimation_radius_search = 10.0 # Adjust based on point density

# Region Growing parameters
min_cluster_size = 100
max_cluster_size = 1000000
number_of_neighbours = 15
smoothness_threshold = 3 / 180.0 * np.pi # in radians
curvature_threshold = 0.8

# --- Main Processing Loop ---
try:
    # Get a list of all .las files in the input directory
    las_files = [f for f in os.listdir(input_las_dir) if f.lower().endswith('.las')]

    if not las_files:
        print(f"No .las files found in the '{input_las_dir}' directory. Please check the path and contents.")
        exit()

    print(f"Found {len(las_files)} .las files in '{input_las_dir}'. Starting region growing for each...")

    for filename in las_files:
        input_file_path = os.path.join(input_las_dir, filename)
        output_filename_base = os.path.splitext(filename)[0] # Get filename without .las extension
        output_pcd_path = os.path.join(output_segmented_dir, f"{output_filename_base}_segmented.pcd")

        print(f"\n--- Processing: {filename} ---")
        try:
            # 1. Load LAS file using pclpy.read
            # pclpy.read takes the path and the point type (e.g., "PointXYZ")
            # For colored output later, we might need PointXYZRGB or ensure colors are handled
            # However, `reg.getColoredCloud()` implies it handles color assignment.
            cloud = pclpy.read(input_file_path, "PointXYZ")
            # print(f"  Loaded point cloud with {cloud.size()} points.")

            # if cloud.size() == 0:
            #     print(f"  File '{filename}' contains no points. Skipping segmentation.")
            #     continue

            # 2. Estimate normals
            print("  Estimating normals...")
            normal_estimation = pcl.features.NormalEstimation.PointXYZ_Normal()
            normals = pcl.PointCloud.Normal()
            tree = pcl.search.KdTree.PointXYZ() # KDTree for normal estimation search
            
            normal_estimation.setInputCloud(cloud)
            normal_estimation.setSearchMethod(tree)
            normal_estimation.setRadiusSearch(normal_estimation_radius_search)
            normal_estimation.compute(normals)
            print(f"  Estimated normals for {normals.size()} points.")

            # 3. Region Growing Segmentation
            print("  Starting region growing segmentation...")
            reg = pcl.segmentation.RegionGrowing.PointXYZ_Normal()
            
            reg.setMinClusterSize(min_cluster_size)
            reg.setMaxClusterSize(max_cluster_size)
            reg.setSearchMethod(tree) # Use the same KDTree
            reg.setNumberOfNeighbours(number_of_neighbours)
            reg.setInputCloud(cloud)
            reg.setInputNormals(normals) # Input normals are crucial for region growing
            reg.setSmoothnessThreshold(smoothness_threshold)
            reg.setCurvatureThreshold(curvature_threshold)
            print("  Parameters set for region growing segmentation.")

            print("  Extracting clusters...")
            clusters = pcl.vectors.PointIndices() # Stores indices of points for each cluster
            reg.extract(clusters)
            print(f"  Extracted {len(clusters)} clusters.")

            if len(clusters) == 0:
                print(f"  No clusters found for '{filename}'. Skipping saving colored cloud.")
                continue

            # 4. Get the colored cloud
            # reg.getColoredCloud() returns a PointCloud<PointXYZRGB> with clusters colored differently
            colored_cloud = reg.getColoredCloud()
            
            # 5. Save the colored point cloud
            pcl.io.savePCDFile(output_pcd_path, colored_cloud)
            print(f"  Successfully saved colored segmented cloud to '{output_pcd_path}'.")

        except Exception as e:
            print(f"  An error occurred while processing '{filename}': {e}. Skipping this file.")

except ImportError:
    print("\n--- CRITICAL ERROR: Library Not Found ---")
    print("The 'pclpy' or 'laspy' library is not installed.")
    print("Please ensure they are installed correctly, especially 'pclpy' (conda install -c conda-forge pclpy).")
    print("-----------------------------------------")
except Exception as e:
    print(f"\nAn unexpected error occurred during overall process: {e}")
    print("Please check your input directories, file permissions, and environment setup.")