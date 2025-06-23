import laspy
import numpy as np
import os
import CSF # Assuming 'cloth-simulation-filter' library is installed

# Define directories
cropped_parts_dir = r"PointClouds\B_cropped_parts"
non_ground_output_dir = r"PointClouds\B_non_ground_parts"

# Create the output directory if it doesn't exist
os.makedirs(non_ground_output_dir, exist_ok=True)
print(f"Ensuring output directory '{non_ground_output_dir}' exists.")

# CSF parameters (as provided in your snippet)
# Note: You might need to tune these for optimal results for your specific data.
# ------ Parameter For PartA ------
# csf_params = {
#     'bSloopSmooth': True,
#     'cloth_resolution': 10.0,   # Grid size of cloth
#     'class_threshold': 100.0,  # Max distance from cloth to be classified as ground
#     'rigidness': 1,            # Rigidity of the cloth (1-3)
#     'iterations': 300          # Max iterations for cloth simulation
# }

# ------ Parameter for PartB ------
csf_params = {
    'bSloopSmooth': True,
    'cloth_resolution': 0.1,   # Grid size of cloth
    'class_threshold': 0.05,  # Max distance from cloth to be classified as ground
    'rigidness': 3,            # Rigidity of the cloth (1-3)
    'iterations': 300          # Max iterations for cloth simulation
}

try:
    # Get a list of all .las files in the cropped_parts directory
    cropped_files = [f for f in os.listdir(cropped_parts_dir) if f.lower().endswith('.las')]

    if not cropped_files:
        print(f"No .las files found in the '{cropped_parts_dir}' directory. Please ensure files exist.")
    else:
        print(f"Found {len(cropped_files)} .las files in '{cropped_parts_dir}'. Starting CSF processing...")

    for filename in cropped_files:
        input_file_path = os.path.join(cropped_parts_dir, filename)
        
        print(f"\n--- Processing: {input_file_path} ---")
        try:
            # 1. Read the LAS file
            las_data = laspy.read(input_file_path)
            # print(f"  Read {las_data.header.number_of_points} points from '{filename}'.")

            # if las_data.header.number_of_points == 0:
            #     print(f"  File '{filename}' contains no points. Skipping.")
            #     continue

            # Get X, Y, Z coordinates for CSF
            xyz_points = np.vstack((las_data.X, las_data.Y, las_data.Z)).transpose()

            # Initialize CSF and set parameters
            csf = CSF.CSF()
            csf.params.bSloopSmooth = csf_params['bSloopSmooth']
            csf.params.cloth_resolution = csf_params['cloth_resolution']
            csf.params.class_threshold = csf_params['class_threshold']
            csf.params.rigidness = csf_params['rigidness']
            csf.params.iterations = csf_params['iterations']

            csf.setPointCloud(xyz_points)

            # Variables to store indices of ground and non-ground points
            ground_indices = CSF.VecInt()
            non_ground_indices = CSF.VecInt()

            # Perform the filtering
            print("  Starting CSF filtering...")
            csf.do_filtering(ground_indices, non_ground_indices)
            print("  CSF filtering complete.")

            # Convert VecInt to NumPy arrays for easier indexing
            non_ground_indices_np = np.array(non_ground_indices)
            print(f"  Identified {len(non_ground_indices_np)} non-ground points.")

            if len(non_ground_indices_np) > 0:
                # Create a new LasData object for only the non-ground points
                new_las = laspy.create(point_format=las_data.header.point_format,
                                       file_version=las_data.header.version)

                # Assign X, Y, Z coordinates for the non-ground points
                new_las.X = las_data.X[non_ground_indices_np]
                new_las.Y = las_data.Y[non_ground_indices_np]
                new_las.Z = las_data.Z[non_ground_indices_np]

                # Copy all other dimensions for the non-ground points
                for dim_name in las_data.point_format.dimension_names:
                    if dim_name not in ['x', 'y', 'z']:
                        setattr(new_las, dim_name, getattr(las_data, dim_name)[non_ground_indices_np])

                # Update header's offsets and scale factors from the original
                # This assumes the input cropped files already have correct zeroed offsets
                new_las.header.x_offset = las_data.header.x_offset
                new_las.header.y_offset = las_data.header.y_offset
                new_las.header.z_offset = las_data.header.z_offset
                new_las.header.x_scale = las_data.header.x_scale
                new_las.header.y_scale = las_data.header.y_scale
                new_las.header.z_scale = las_data.header.z_scale

                # Update the new header's min/max bounds based on the actual non-ground data
                new_las.header.x_min = np.min(new_las.X)
                new_las.header.x_max = np.max(new_las.X)
                new_las.header.y_min = np.min(new_las.Y)
                new_las.header.y_max = np.max(new_las.Y)
                new_las.header.z_min = np.min(new_las.Z)
                new_las.header.z_max = np.max(new_las.Z)

                # Define the output file path for the non-ground points
                output_filename = f"non_ground_{filename}"
                output_file_path = os.path.join(non_ground_output_dir, output_filename)

                # Write the new LasData object to the output file
                print(f"  Saving non-ground points to: '{output_filename}'")
                new_las.write(output_file_path)
                print(f"  Successfully saved non-ground points from '{filename}'.")
            else:
                print(f"  No non-ground points found for '{filename}'. No file saved for this part.")

        except FileNotFoundError:
            print(f"  Error: Input file '{input_file_path}' not found. Skipping.")
        except Exception as e:
            print(f"  An error occurred while processing '{filename}': {e}. Skipping.")

except ImportError:
    print("\n--- CRITICAL ERROR: Library Not Found ---")
    print("The 'laspy' or 'CSF' (cloth-simulation-filter) library is not installed.")
    print("Please install them using:")
    print("pip install laspy numpy open3d matplotlib")
    print("pip install cloth-simulation-filter")
    print("-----------------------------------------")
except Exception as e:
    print(f"\nAn unexpected error occurred during overall process: {e}")
    print("Please check your input directories and file permissions.")