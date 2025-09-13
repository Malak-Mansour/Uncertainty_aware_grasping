
'''
    mean_out2 (MC dropout mean of the points: can be visualized)
    xyz (final output prediction: can be visualized)
    std_out2 (uncertainty std values of all the points. RMS of the x,y,z uncertainty to process it when grasping)
    src_pcd (Source Partial Input)
    src_pcd_inter (Intermediate Output)
    model_pcd_transformed (ground truth)
'''

import numpy as np
import os
import open3d as o3d

# List of source paths
src_paths = [
    "/home/malak.mansour/Downloads/PointAttN-Modified/PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_no_occ_cd_debug_pcn/all_no_dropout/batch0_sample0_data.npz",
    "/home/malak.mansour/Downloads/PointAttN-Modified/PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.1_cd_debug_pcn/all_no_dropout/batch0_sample0_data.npz",
    "/home/malak.mansour/Downloads/PointAttN-Modified/PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.2_cd_debug_pcn/all_no_dropout/batch0_sample0_data.npz",
    "/home/malak.mansour/Downloads/PointAttN-Modified/PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.3_cd_debug_pcn/all_no_dropout/batch0_sample0_data.npz",
    "/home/malak.mansour/Downloads/PointAttN-Modified/PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.4_cd_debug_pcn/all_no_dropout/batch0_sample0_data.npz",

    "/home/malak.mansour/Downloads/PointAttN-Modified/PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_no_occ_cd_debug_pcn/all_dropout/batch0_sample0_data.npz",
    "/home/malak.mansour/Downloads/PointAttN-Modified/PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.1_cd_debug_pcn/all_dropout/batch0_sample0_data.npz",   
    "/home/malak.mansour/Downloads/PointAttN-Modified/PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.2_cd_debug_pcn/all_dropout/batch0_sample0_data.npz",   
    "/home/malak.mansour/Downloads/PointAttN-Modified/PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.3_cd_debug_pcn/all_dropout/batch0_sample0_data.npz",   
    "/home/malak.mansour/Downloads/PointAttN-Modified/PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.4_cd_debug_pcn/all_dropout/batch0_sample0_data.npz",   
]

dst_folder = 'npy_files'


def clean_pointcloud(pc_full, nb_neighbors=30, std_ratio=0.5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_full)
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return np.asarray(pcd_clean.points)


# Process all files
for src in src_paths:

    # Load the npz file
    data = np.load(src)

    pc_full = data['xyz']

    pc_denoised = clean_pointcloud(pc_full)
    if pc_denoised.shape[0] < 20:
        print("[WARN] Too few points after filtering, using raw pc_full")
        pc_denoised = pc_full


    # Extract the folder name 2 levels above the file
    folder_name = os.path.basename(os.path.dirname(os.path.dirname(src)))

    # Create the destination filename using the folder name
    dst_filename = f"{folder_name}.npy"
    dst = os.path.join(dst_folder, dst_filename)

    # Create dst_folder if it doesn't exist
    os.makedirs(dst_folder, exist_ok=True)

    # Save as npy file
    # np.save(dst, pc_full)
    
    # Save as npy file (denoised instead of raw)
    np.save(dst, pc_denoised)
    