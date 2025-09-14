import numpy as np
import os
import glob
import open3d as o3d

# List of source paths (same as before)
src_paths = [
    "../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_no_occ_cd_debug_pcn/all_no_dropout/*.npz",
    "../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.1_cd_debug_pcn/all_no_dropout/*.npz",
    "../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.2_cd_debug_pcn/all_no_dropout/*.npz",
    "../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.3_cd_debug_pcn/all_no_dropout/*.npz",
    "../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.4_cd_debug_pcn/all_no_dropout/*.npz",

    "../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_no_occ_cd_debug_pcn/all_dropout/*.npz",
    "../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.1_cd_debug_pcn/all_dropout/*.npz",
    "../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.2_cd_debug_pcn/all_dropout/*.npz",
    "../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.3_cd_debug_pcn/all_dropout/*.npz",
    "../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.4_cd_debug_pcn/all_dropout/*.npz",
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
for src_pattern in src_paths:
    # Get all npz files matching the pattern
    for src_file in glob.glob(src_pattern):
        try:
            # Load the npz file
            data = np.load(src_file)
            pc_full = data['xyz']

            pc_denoised = clean_pointcloud(pc_full)
            if pc_denoised.shape[0] < 20:
                print(f"[WARN] Too few points after filtering for {src_file}, using raw pc_full")
                pc_denoised = pc_full

            # Extract the original filename without extension
            base_filename = os.path.splitext(os.path.basename(src_file))[0]
            # Get the folder name 2 levels above
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(src_file)))
            
            # Create destination filename
            dst_filename = f"{folder_name}_{base_filename}.npy"
            dst = os.path.join(dst_folder, dst_filename)

            # Create dst_folder if it doesn't exist
            os.makedirs(dst_folder, exist_ok=True)

            # Save as npy file
            np.save(dst, pc_denoised)
            print(f"Processed: {src_file} -> {dst}")
            
        except Exception as e:
            print(f"Error processing {src_file}: {str(e)}")
