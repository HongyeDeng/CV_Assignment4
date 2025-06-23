import laspy
import numpy as np
import os

# Define the directory containing the .las files to combine
input_parts_dir = r"PointClouds\B_smooth_non_ground_parts" # Or "smooth_non_ground_parts" or "cropped_parts"
# Define the path and name for the single output .las file
output_combined_file_path = r"PointClouds\IntermediateFile_B\B_smooth_non_ground.las"

try:
    # Get a list of all .las files in the input directory
    las_files_to_combine = [f for f in os.listdir(input_parts_dir) if f.lower().endswith('.las')]

    if not las_files_to_combine:
        print(f"No .las files found in the '{input_parts_dir}' directory. Exiting.")
        exit()

    print(f"Found {len(las_files_to_combine)} .las files in '{input_parts_dir}'.")

    # Initialize a list to hold LasData objects or point data from each file
    all_points_data_list = []
    
    # Store the header of the first file to use as a template for the combined file
    # This assumes all input files have a compatible point format and version.
    base_header = None 
    base_point_format = None

    # Loop through each .las file, read its data, and collect points
    for i, filename in enumerate(las_files_to_combine):
        current_file_path = os.path.join(input_parts_dir, filename)
        print(f"--- Reading {i+1}/{len(las_files_to_combine)}: {filename} ---")
        
        try:
            las_data = laspy.read(current_file_path)
            # num_points = las_data.points.count
            # print(f"  Loaded {num_points} points.")

            # if num_points == 0:
            #     print(f"  File '{filename}' contains no points. Skipping.")
            #     continue
            
            # Store the first file's header and point format as our template
            if base_header is None:
                base_header = las_data.header
                base_point_format = las_data.header.point_format
                print("  Using first file's header as template for combined file.")

            # Create a dictionary for current file's point data, including all dimensions
            current_file_points = {}
            for dim_name in las_data.point_format.dimension_names:
                # Use .array to get the raw (unscaled) integer data
                current_file_points[dim_name] = getattr(las_data, dim_name).copy() 
            
            all_points_data_list.append(current_file_points)

        except Exception as e:
            print(f"  An error occurred while reading '{filename}': {e}. Skipping this file.")
            continue

    if not all_points_data_list:
        print("No valid point clouds were read. No combined file will be created. Exiting.")
        exit()

    print("\n--- Combining point clouds ---")

    # Determine the total number of points and dimensions needed for the combined file
    # This part can be tricky if input files have different dimensions or order.
    # We assume consistent point formats for now based on previous processing steps.
    
    # Initialize lists to store concatenated data for each dimension
    combined_dimension_data = {dim.name: [] for dim in base_point_format.dimensions}
    
    for file_data in all_points_data_list:
        for dim_name, data_array in file_data.items():
            if dim_name in combined_dimension_data: # Ensure dimension exists in our base format
                combined_dimension_data[dim_name].append(data_array)
            # else: Handle cases where a dimension might be missing in some files (e.g., fill with default)

    # Concatenate all arrays for each dimension
    final_point_count = 0
    for dim_name in combined_dimension_data:
        if combined_dimension_data[dim_name]: # Only concatenate if there's data for this dim
            combined_dimension_data[dim_name] = np.concatenate(combined_dimension_data[dim_name])
            if final_point_count == 0:
                final_point_count = len(combined_dimension_data[dim_name])
        else:
            # If a dimension ended up empty, ensure it's an empty array matching expected count
            combined_dimension_data[dim_name] = np.array([], dtype=int) # Or appropriate dtype
            print(f"Warning: No data found for dimension '{dim_name}' across all files. It will be empty in output.")

    print(f"Total points to be combined: {final_point_count}")

    if final_point_count == 0:
        print("Combined dataset is empty. No file will be saved. Exiting.")
        exit()

    # Create the new LasData object for the combined data
    # Use the base_header's point format and version
    combined_las = laspy.create(point_format=base_point_format,
                                file_version=base_header.version)

    # Assign concatenated data to the new LasData object's dimensions
    for dim_name, data_array in combined_dimension_data.items():
        # laspy 2.x uses capitalized attributes for scaled coordinates (X,Y,Z)
        # but internally it maps 'x' (lowercase) to the actual data.
        # getattr and setattr on las_data directly use the lowercase names from point_format.dimension_names.
        # However, to be safe and consistent with previous examples, if we extracted las_data.X etc.
        # we should assign to new_las.X, new_las.Y, new_las.Z
        if dim_name.lower() in ['x', 'y', 'z']:
            setattr(combined_las, dim_name.upper(), data_array) # Assign to X, Y, Z
        else:
            setattr(combined_las, dim_name, data_array) # Assign other dimensions

    # Set header offsets to zero for a local coordinate system if input files were also zeroed
    # This assumes the individual parts already had their offsets properly handled.
    # If not, laspy will try to compute optimal offsets on write unless explicitly set.
    combined_las.header.x_offset = base_header.x_offset
    combined_las.header.y_offset = base_header.y_offset
    combined_las.header.z_offset = base_header.z_offset

    # Copy scale factors from the base header
    combined_las.header.x_scale = base_header.x_scale
    combined_las.header.y_scale = base_header.y_scale
    combined_las.header.z_scale = base_header.z_scale

    # Update global min/max for the combined dataset in the header
    # laspy will automatically compute these on write if not set, but explicit is good.
    # Need to use the _actual_ floating point values (new_las.X) for min/max calculation
    # if offsets are zeroed, these are directly the min/max of the assigned X/Y/Z arrays.
    combined_las.header.x_min = np.min(combined_las.X)
    combined_las.header.x_max = np.max(combined_las.X)
    combined_las.header.y_min = np.min(combined_las.Y)
    combined_las.header.y_max = np.max(combined_las.Y)
    combined_las.header.z_min = np.min(combined_las.Z)
    combined_las.header.z_max = np.max(combined_las.Z)


    # Save the combined LasData object to the new .las file
    print(f"\nSaving combined point cloud to: '{output_combined_file_path}'")
    combined_las.write(output_combined_file_path)
    print(f"Successfully combined {final_point_count} points into '{output_combined_file_path}'")

except FileNotFoundError:
    print(f"Error: The input directory '{input_parts_dir}' not found.")
    print("Please ensure the directory exists and contains .las files.")
except ImportError:
    print("\n--- CRITICAL ERROR: Library Not Found ---")
    print("The 'laspy' library is not installed.")
    print("Please install it using: pip install laspy numpy")
    print("-----------------------------------------")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    print("Please check your input files, directory paths, and environment setup.")