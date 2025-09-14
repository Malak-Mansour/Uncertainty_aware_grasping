import numpy as np
import os

# === CONFIG ===
src_paths = [
    "../../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_no_occ_cd_debug_pcn/all_no_dropout/batch0_sample0_data.npz",
    "../../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.1_cd_debug_pcn/all_no_dropout/batch0_sample0_data.npz",
    "../../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.2_cd_debug_pcn/all_no_dropout/batch0_sample0_data.npz",
    "../../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.3_cd_debug_pcn/all_no_dropout/batch0_sample0_data.npz",
    "../../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_occ_0.4_cd_debug_pcn/all_no_dropout/batch0_sample0_data.npz",

    "../../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_no_occ_cd_debug_pcn/all_dropout/batch0_sample0_data.npz",
    "../../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.1_cd_debug_pcn/all_dropout/batch0_sample0_data.npz",
    "../../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.2_cd_debug_pcn/all_dropout/batch0_sample0_data.npz",
    "../../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.3_cd_debug_pcn/all_dropout/batch0_sample0_data.npz",
    "../../PointAttN-Modified_uncertainty/log/PointAttN_baseline_cd_matching_f1_MC_occ_0.4_cd_debug_pcn/all_dropout/batch0_sample0_data.npz",
]

results_folder = "results"
new_results_folder = "results_with_completion_fields"
os.makedirs(new_results_folder, exist_ok=True)

# === PROCESS ===
for src in src_paths:
    folder_name = os.path.basename(os.path.dirname(os.path.dirname(src)))
    dst_filename = f"predictions_{folder_name}.npz"
    dst_path = os.path.join(results_folder, dst_filename)

    if not os.path.exists(dst_path):
        print(f"[WARN] No matching results file for {folder_name}, skipping.")
        continue

    print(f"[INFO] Creating augmented results for {dst_filename}")

    # Load source fields
    src_data = np.load(src, allow_pickle=True)
    new_fields = {
        "mean_out2": src_data["mean_out2"],
        "xyz": src_data["xyz"],
        "std_out2": src_data["std_out2"],
        "src_pcd": src_data["src_pcd"],
        "src_pcd_inter": src_data["src_pcd_inter"],
        "model_pcd_transformed": src_data["model_pcd_transformed"],
    }

    # Load existing results file
    dst_data = np.load(dst_path, allow_pickle=True)
    combined_fields = {key: dst_data[key] for key in dst_data.files}
    combined_fields.update(new_fields)

    # Save to new folder
    new_path = os.path.join(new_results_folder, dst_filename)
    np.savez(new_path, **combined_fields)

print(f"[DONE] Augmented files saved to: {new_results_folder}")
