import pclpy
from pclpy import pcl
import laspy
import numpy as np
import os

# Step 1: Load LAS file
las_path = r"PointClouds\smooth_non_ground_PartB.las"
las = pclpy.read(las_path, "PointXYZ")

# Step 2: Create PointCloud object and fill with data
cloud = las
print(f"Loaded point cloud with {cloud.size()} points.")

# Step 3: Estimate normals
print("Estimating normals...")
normal_estimation = pcl.features.NormalEstimation.PointXYZ_Normal()
normals = pcl.PointCloud.Normal()
tree = pcl.search.KdTree.PointXYZ()
normal_estimation.setInputCloud(cloud)
normal_estimation.setSearchMethod(tree)
normal_estimation.setRadiusSearch(10)  # adjust based on point density
normal_estimation.compute(normals)
print(f"Estimated normals for {normals.size()} points.")

# Step 4: Region Growing Segmentation
print("Starting region growing segmentation...")
reg = pcl.segmentation.RegionGrowing.PointXYZ_Normal()
reg.setMinClusterSize(50)
reg.setMaxClusterSize(1000000)
reg.setSearchMethod(tree)
reg.setNumberOfNeighbours(5)
reg.setInputCloud(cloud)
reg.setInputNormals(normals)
reg.setSmoothnessThreshold(3 / 180.0 * np.pi)  # in radians
reg.setCurvatureThreshold(0.5)
print("Parameters set for region growing segmentation.")

print("Extracting clusters...")
clusters = pcl.vectors.PointIndices()
reg.extract(clusters)
print(f"Extracted {len(clusters)} clusters.")

print(f"Found {len(clusters)} clusters")

# Step 5: Color the clusters and save
colored_cloud = reg.getColoredCloud()
print(type(colored_cloud))

colored_cloud = pcl.PointCloud.PointXYZRGB()
colored_cloud = reg.getColoredCloud()
pcl.io.savePCDFile("region_growing_colored_B.pcd", colored_cloud)