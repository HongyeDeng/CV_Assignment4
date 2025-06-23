import laspy
import numpy as np
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm   
import CSF

pointcloud_file_path = r'PointClouds\cropped_PartA.las'
las = laspy.read(pointcloud_file_path)
print("File read successfully.")
# 2. Extract coordinates (X, Y, Z)
# las.X, las.Y, las.Z provide the scaled (true) coordinates as floats
points = np.vstack((las.X, las.Y, las.Z)).transpose()
print(f"Extracted {points.shape[0]} points.")

csf = CSF.CSF()

#------ Parameters for PartA --------
csf.params.bSloopSmooth = True        # Set to True for smoother results on steep slopes
csf.params.cloth_resolution = 10    # Grid size of cloth. Smaller for finer detail.
csf.params.class_threshold = 5      # Max distance from cloth to be classified as ground
csf.params.rigidness = 3              # Rigidity of the cloth (1-3, 1 for soft, 3 for rigid)
csf.params.iterations = 300           # Max iterations for cloth simulation

#------ Parameters for PartB --------
# csf.params.bSloopSmooth = True        # Set to True for smoother results on steep slopes
# csf.params.cloth_resolution = 0.1    # Grid size of cloth. Smaller for finer detail.
# csf.params.class_threshold = 0.01      # Max distance from cloth to be classified as ground
# csf.params.rigidness = 3              # Rigidity of the cloth (1-3, 1 for soft, 3 for rigid)
# csf.params.iterations = 300           # Max iterations for cloth simulation

csf.setPointCloud(points)
# Variables to store indices of ground and non-ground points
ground_indices = CSF.VecInt()
non_ground_indices = CSF.VecInt()

# Perform the filtering
print("Starting CSF filtering...")
csf.do_filtering(ground_indices, non_ground_indices)
print("CSF filtering complete.")

# Convert VecInt to NumPy arrays for easier indexing
ground_indices_np = np.array(ground_indices)
non_ground_indices_np = np.array(non_ground_indices)

print(f"Number of ground points: {len(ground_indices_np)}")
print(f"Number of non-ground points: {len(non_ground_indices_np)}")

# 3. Create Open3D PointCloud objects for visualization
# Assign different colors for ground and non-ground points

# Ground points (Green)
ground_pcd = o3d.geometry.PointCloud()
if len(ground_indices_np) > 0:
    ground_pcd.points = o3d.utility.Vector3dVector(points[ground_indices_np])
    ground_pcd.paint_uniform_color([0, 1, 0])  # Green for ground points
else:
    print("No ground points found by CSF or an issue occurred during filtering.")

# Non-ground points (Red - for buildings/vegetation)
non_ground_pcd = o3d.geometry.PointCloud()
if len(non_ground_indices_np) > 0:
    non_ground_pcd.points = o3d.utility.Vector3dVector(points[non_ground_indices_np])
    non_ground_pcd.paint_uniform_color([1, 0, 0])  # Red for non-ground points
else:
    print("No non-ground points found by CSF or an issue occurred during filtering.")


# 4. Visualize the point clouds with Open3D
print("Opening Open3D visualization window...")
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Ground (Green) and Non-Ground (Red) Points (CSF Filtered)", width=1024, height=768)

render_option = vis.get_render_option()
render_option.point_size = 0.5 # Adjust point size as desired for visualization
# You can also set a background color if preferred:
# render_option.background_color = np.asarray([0.1, 0.1, 0.1])

if len(ground_indices_np) > 0:
    vis.add_geometry(ground_pcd)
if len(non_ground_indices_np) > 0:
    vis.add_geometry(non_ground_pcd)

vis.run() # This starts the interactive visualization loop
vis.destroy_window()
print("Visualization closed.")
