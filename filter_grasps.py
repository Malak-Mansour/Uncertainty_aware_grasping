# !pip install trimesh torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

'''
    pc_full: (21807, 3)
    pred_grasps_cam: ()
    scores: ()
    contact_pts: ()
    pc_colors: ()
    
    
    mean_out2 (MC dropout mean of the points: can be visualized)
    xyz (final output prediction: can be visualized)
    std_out2 (uncertainty std values of all the points. RMS of the x,y,z uncertainty to process it when grasping)
    src_pcd (Source Partial Input)
    src_pcd_inter (Intermediate Output)
    model_pcd_transformed (ground truth)
'''


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d as o3d
import numpy as np
import glob
import os
from viz_functions import visualize_grasps  


def load_pointcloud_stddev(npz_data):
    """
    Load per-point stddevs directly from already-merged results_with_completion_fields file.
    Falls back to zeros if std_out2 not present.
    """
    if "std_out2" not in npz_data:
        print("[WARN] std_out2 missing, assuming uncertainty=0")
        sigma = np.zeros(npz_data["pc_full"].shape[0])
        return sigma, 0.0

    std_xyz = npz_data["std_out2"]   # (N,3)
    sigma = np.sqrt((std_xyz**2).mean(axis=1))   # RMS across xyz → (N,)
    mean_sigma = sigma.mean()
    return sigma, mean_sigma


def filter_grasps_with_uncertainty(pc_full, pred_grasps_cam, scores, 
                                   sigma, mean_sigma,
                                   front_axis=np.array([0, 0, 1]), 
                                   dot_thresh=0.7, 
                                   point_stddev_thresh=0.01, 
                                   global_stddev_thresh=0.05,
                                   k_neighbors=10,
                                   topk=10):
    print("="*60)
    print(f"[DEBUG] Starting filtering...")
    print(f"[DEBUG] pc_full shape={pc_full.shape}, grasps={len(pred_grasps_cam.get(-1, []))}")
    print(f"[DEBUG] sigma size={sigma.shape}, mean_sigma={mean_sigma:.4f}, global threshold={global_stddev_thresh}")

    if mean_sigma > global_stddev_thresh:
        print("[INFO] ❌ Strawberry too uncertain globally, no grasps possible.")
        return None, None

    if -1 not in pred_grasps_cam:
        print("[WARN] No grasps found in dict.")
        return None, None

    grasps = pred_grasps_cam[-1]
    conf   = scores[-1]

    valid_idx = []
    for i, g in enumerate(grasps):
        approach = g[:3, 2] / np.linalg.norm(g[:3, 2])
        dot = np.dot(approach, front_axis)

        center = g[:3, 3]
        dists = np.linalg.norm(pc_full - center, axis=1)
        knn_idx = np.argsort(dists)[:k_neighbors]
        local_std = sigma[knn_idx].mean() if sigma.size > 1 else 0.0

        print(f"[DEBUG] grasp {i:03d}:")
        print(f"        center={center}, approach={approach}")
        print(f"        dot(front_axis)={dot:.3f} vs thresh={dot_thresh}")
        print(f"        local_std={local_std:.4f} vs thresh={point_stddev_thresh}")
        print(f"        score={conf[i]:.4f}")

        if dot >= dot_thresh and local_std <= point_stddev_thresh:
            print("        -> ✅ PASSED")
            valid_idx.append(i)
        else:
            print("        -> ❌ REJECTED")

    if not valid_idx:
        print("[INFO] ❌ No grasps passed filtering.")
        return None, None

    valid_scores = conf[valid_idx]
    topk_idx = np.argsort(valid_scores)[-topk:]
    topk_grasps = grasps[valid_idx][topk_idx]
    topk_scores = valid_scores[topk_idx]

    print(f"[INFO] ✅ {len(topk_idx)} grasps kept after filtering")
    for j, s in enumerate(topk_scores):
        print(f"   Kept grasp #{j} with score={s:.4f}")

    return {-1: topk_grasps}, {-1: topk_scores}


grasp_files = glob.glob("predicted_grasps_GraspNet/results_with_completion_fields/predictions_PointAttN_baseline_cd_matching_f1_*.npz")

for f in grasp_files:
    print("Visualizing", f)
    data = np.load(f, allow_pickle=True)

    pc_full   = data["pc_full"]
    pc_colors = data["pc_colors"]
    pred_grasps_cam = data["pred_grasps_cam"].item()
    scores    = data["scores"].item()


    sigma, mean_sigma = load_pointcloud_stddev(data)




    # --- Map sigma → colors using a colormap ---
    norm_sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-8)  # normalize 0–1
    cmap = cm.get_cmap("jet")   # colormap: jet, viridis, plasma, etc.
    pc_uncertainty_colors = cmap(norm_sigma)[:, :3]  # take RGB (ignore alpha)
    print(f"[DEBUG] Uncertainty visualization for {f}")
    print(f"    sigma: min={sigma.min():.4f}, max={sigma.max():.4f}, mean={mean_sigma:.4f}")

    # Show uncertainty-colored point cloud
    # o3d.visualization.draw_geometries([pcd])




    # Add colorbar legend for uncertainty
    fig, ax = plt.subplots(figsize=(5, 1))
    cb = plt.colorbar(
        cm.ScalarMappable(cmap=cmap),
        cax=ax, orientation="horizontal"
    )
    cb.set_label("Uncertainty")
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels([f"{sigma.min():.3f}", f"{mean_sigma:.3f}", f"{sigma.max():.3f}"])
    plt.show()




    # Realsense camera axes
        # +X → to the right (camera’s right)
        # +Y → down (camera’s bottom)
        # +Z → forward (out of the camera, into the scene)
    centroid = pc_full.mean(axis=0)
    T_strawberry = np.array([
        [1, 0, 0, centroid[0]],
        [0, -1, 0, centroid[1]],
        [0, 0, 1, centroid[2]],
        [0, 0, 0, 1]
    ])
    front_axis = T_strawberry[:3, 2] / np.linalg.norm(T_strawberry[:3, 2])  # Z-axis (forward)



    filtered_grasps, filtered_scores = filter_grasps_with_uncertainty(
        pc_full, pred_grasps_cam, scores,
        sigma=sigma, mean_sigma=mean_sigma,
        front_axis=front_axis, 
        dot_thresh=0.7,
        point_stddev_thresh=0.01,
        global_stddev_thresh=0.05,
        k_neighbors=10,
        topk=10
    )

    if filtered_grasps is not None:
        visualize_grasps(
            pc_full,
            filtered_grasps,
            filtered_scores,
            # pc_colors=pc_colors,
            pc_colors=(pc_uncertainty_colors*255),
            plot_others=[T_strawberry]
        )
