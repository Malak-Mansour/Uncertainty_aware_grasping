import numpy as np
import os
import open3d as o3d

def create_leaf_occlusion(points, leaf_coverage=0.2, leaf_thickness=0.01):
    """
    Apply a side leaf-shaped occlusion to the strawberry point cloud.

    Args:
        points (Nx3): input point cloud
        leaf_coverage (float): relative size of leaf footprint (0-1)
        leaf_thickness (float): thickness of leaf occluder in direction normal to leaf plane
    """
    occluded_points = points.copy()
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    diag_size = np.linalg.norm(max_coords - min_coords)

    # Randomly choose which side the leaf attaches to (x or y direction, ±)
    side_axis = np.random.choice([0, 1])  # 0 -> X, 1 -> Y
    side_sign = np.random.choice([-1, 1])

    # Leaf center near the side
    leaf_center = np.array([
        max_coords[0] if (side_axis == 0 and side_sign > 0) else 
        min_coords[0] if (side_axis == 0) else (min_coords[0] + max_coords[0]) / 2,
        
        max_coords[1] if (side_axis == 1 and side_sign > 0) else 
        min_coords[1] if (side_axis == 1) else (min_coords[1] + max_coords[1]) / 2,
        
        (min_coords[2] + max_coords[2]) / 2  # around middle in Z
    ])

    # Normal: pointing inward (from outside toward center)
    normal = np.zeros(3)
    normal[side_axis] = -side_sign

    # Axes spanning the leaf plane (perpendicular to normal)
    if side_axis == 0:  # leaf in Y-Z plane
        axis1, axis2 = np.array([0, 1, 0]), np.array([0, 0, 1])
    else:               # leaf in X-Z plane
        axis1, axis2 = np.array([1, 0, 0]), np.array([0, 0, 1])

    # Ellipse footprint (leaf shape elongated)
    major = diag_size * leaf_coverage
    minor = major * 0.4  # thinner for leaf-like look

    # Project points into leaf coordinate system
    rel = occluded_points - leaf_center
    d_normal = rel @ normal
    u = rel @ axis1
    v = rel @ axis2

    # Footprint + thickness check
    inside_leaf = (u / major) ** 2 + (v / minor) ** 2 <= 1
    inside_thickness = np.abs(d_normal) <= leaf_thickness
    mask = ~(inside_leaf & inside_thickness)

    occluded_points = occluded_points[mask]
    return occluded_points


if __name__ == "__main__":
    in_dir = "random_ply"   # folder containing original .npz
    out_dir = "occluded"
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(in_dir):
        if file.endswith(".npz"):
            file_path = os.path.join(in_dir, file)
            data = np.load(file_path)

            if "src_pcd" in data and len(data["src_pcd"]) > 0:
                original = data["src_pcd"]

                # Apply leaf occlusion
                occluded = create_leaf_occlusion(original, leaf_coverage=0.4, leaf_thickness=0.5)

                print(f"{file}: Original {len(original)} pts → Occluded {len(occluded)} pts")

                base = os.path.splitext(file)[0]
                out_path = os.path.join(out_dir, base + "_leaf.npz")
                np.savez(out_path, src_pcd=occluded)

    print("✅ All files processed with random leaf occlusions from the SIDES")


# --- Visualization (optional) ---
directory = "occluded"
for file in os.listdir(directory):
    if file.endswith('.npz'):
        file_path = os.path.join(directory, file)
        data = np.load(file_path)

        if 'src_pcd' in data:
            src_pcd = data['src_pcd']
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(src_pcd)
            point_cloud.paint_uniform_color([0, 1, 0])  # green

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"{file} - src_pcd")
            vis.add_geometry(point_cloud)
            vis.run()
            vis.destroy_window()
