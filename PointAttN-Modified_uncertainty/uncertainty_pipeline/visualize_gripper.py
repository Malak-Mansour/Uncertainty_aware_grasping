import json
import numpy as np
import open3d as o3d

# --- Load from .npz ---
data = np.load("log/PointAttN_baseline_cd_matching_f1_cd_debug_pcn/all/batch2_sample12_data.npz")
P = data["mean_out2"]        # (N,3) completed cloud
sigma = data["std_out2"]     # (N,3) or (N,)
# optional coloring by uncertainty
if sigma.ndim == 2:
    sigma = np.sqrt((sigma**2).mean(axis=1))  # collapse to (N,)
s = (sigma - sigma.min()) / (np.ptp(sigma) + 1e-8)
colors = np.stack([s, 1 - s, np.zeros_like(s)], axis=1)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(P)
pcd.colors = o3d.utility.Vector3dVector(colors)

# --- Load grasp pose ---
with open("grasp_pose.json", "r") as f:
    grasp = json.load(f)

t = np.array(grasp["t"])       # (3,)
R = np.array(grasp["R"])       # (3,3)
w = grasp["width"]             # jaw opening width

# --- Draw gripper frame ---
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
frame.translate(t)
frame.rotate(R, center=t)

# --- Draw jaws as boxes ---
jaw_height = 0.02   # 2 cm
jaw_depth  = 0.005  # 5 mm thickness
half_w = w / 2

left_center = t - half_w * R[:,0]
right_center = t + half_w * R[:,0]

def make_jaw(center, R, size=(0.005, 0.005, 0.02)):
    box = o3d.geometry.TriangleMesh.create_box(*size)
    box.paint_uniform_color([0, 0, 1])  # blue jaws
    box.translate(-box.get_center())    # center at origin
    box.rotate(R, center=(0,0,0))
    box.translate(center)
    return box

jaw_left = make_jaw(left_center, R)
jaw_right = make_jaw(right_center, R)

# --- Visualize ---
o3d.visualization.draw_geometries([pcd, frame, jaw_left, jaw_right])
