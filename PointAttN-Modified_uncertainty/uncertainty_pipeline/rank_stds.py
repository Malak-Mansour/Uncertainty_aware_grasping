import os
import numpy as np
import glob
import open3d as o3d
import matplotlib.pyplot as plt



def visualize_prediction_with_gt_and_input(
    pred_xyz_mc=None,
    pred_xyz_single=None,
    gt_xyz=None,
    partial_xyz=None,
    title="",
    show_mc=True,
    show_single=True,
    show_gt=True,
    show_input=True
):
    geoms = []

    if show_mc and pred_xyz_mc is not None:
        pcd_mc = o3d.geometry.PointCloud()
        pcd_mc.points = o3d.utility.Vector3dVector(pred_xyz_mc)
        pcd_mc.paint_uniform_color([1, 0, 0])  # Red
        geoms.append(pcd_mc)

    if show_single and pred_xyz_single is not None:
        pcd_single = o3d.geometry.PointCloud()
        pcd_single.points = o3d.utility.Vector3dVector(pred_xyz_single)
        pcd_single.paint_uniform_color([1, 0.5, 0])  # Orange
        geoms.append(pcd_single)

    if show_gt and gt_xyz is not None:
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_xyz)
        pcd_gt.paint_uniform_color([0, 1, 0])  # Green
        geoms.append(pcd_gt)

    if show_input and partial_xyz is not None:
        pcd_input = o3d.geometry.PointCloud()
        pcd_input.points = o3d.utility.Vector3dVector(partial_xyz)
        pcd_input.paint_uniform_color([0, 0.5, 1])  # Blue
        geoms.append(pcd_input)

    if len(geoms) == 0:
        print("⚠️ Nothing selected for visualization.")
        return

    o3d.visualization.draw_geometries(geoms, window_name=title)


def rank_strawberries(npz_folder):
    all_files = sorted(glob.glob(os.path.join(npz_folder, '*.npz')))
    rankings = []

    for f in all_files:
        data = np.load(f)

        std_out2 = data['std_out2']  # (N, 3)
        if std_out2.ndim == 2:
            per_point_std = np.linalg.norm(std_out2, axis=1)  # Shape (N,)
        else:
            per_point_std = std_out2.squeeze()

        overall_std = per_point_std.mean()
        rankings.append((f, overall_std))

    rankings.sort(key=lambda x: x[1])  # Ascending = Most certain first
    return rankings

# folder = 'log/PointAttN_baseline_cd_matching_f1_cd_debug_pcn/all'   #batch2_sample12_data.npz - Mean STD: 0.004083   
folder = 'log/PointAttN_baseline_cd_matching_f1_MC_cd_debug_pcn/all_T60_0.1' #batch1_sample13_data.npz - Mean STD: 0.001337
rankings = rank_strawberries(folder)

print("\n🍓 Ranked Predictions by Certainty (lower = more confident):")
for i, (f, score) in enumerate(rankings):
    print(f"{i+1:2d}. {os.path.basename(f)} - Mean STD: {score:.6f}")

# Visualize most and least certain
most_certain_file = rankings[0][0]
least_certain_file = rankings[-1][0]

# most_certain_file = "./all_T60_0.1train_0inference/batch1_sample13_data.npz"
# least_certain_file = "./all_T60_0.1train_0inference/batch2_sample2_data.npz"


data = np.load(most_certain_file)
# visualize_prediction_with_gt_and_input(
#     pred_xyz_mc=data['mean_out2'],
#     pred_xyz_single=data['xyz'],
#     gt_xyz=data['model_pcd_transformed'],
#     partial_xyz=data['src_pcd'],
#     title="Most Certain Prediction",
#     show_mc=True,
#     show_single=True,
#     show_gt=True,
#     show_input=True
# )


data = np.load(least_certain_file)
# visualize_prediction_with_gt_and_input(
#     pred_xyz_mc=data['mean_out2'],
#     pred_xyz_single=data['xyz'],
#     gt_xyz=data['model_pcd_transformed'],
#     partial_xyz=data['src_pcd'],
#     title="Least Certain Prediction",
#     show_mc=True,
#     show_single=True,
#     show_gt=True,
#     show_input=True
# )