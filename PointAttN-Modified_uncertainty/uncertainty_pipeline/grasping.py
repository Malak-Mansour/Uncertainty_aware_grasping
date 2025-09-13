# import os
# import numpy as np
# import glob

# def rank_strawberries(npz_folder):
#     all_files = sorted(glob.glob(os.path.join(npz_folder, '*.npz')))
#     rankings = []

#     for f in all_files:
#         data = np.load(f)

#         std_out2 = data['std_out2']  # (N, 3)
#         if std_out2.ndim == 2:
#             per_point_std = np.linalg.norm(std_out2, axis=1)  # Shape (N,)
#         else:
#             per_point_std = std_out2.squeeze()

#         overall_std = per_point_std.mean()
#         rankings.append((f, overall_std))

#     rankings.sort(key=lambda x: x[1])  # Ascending = Most certain first
#     return rankings


# folder = 'all_T60_0.1' #'all_0.1_sim_train_NN'  
# rankings = rank_strawberries(folder)

# print("\n🍓 Ranked Predictions by Certainty (lower = more confident):")
# for i, (f, score) in enumerate(rankings):
#     print(f"{i+1:2d}. {os.path.basename(f)} - Mean STD: {score:.6f}")

# # Visualize most and least certain
# most_certain_file = rankings[0][0]
# least_certain_file = rankings[-1][0]




# import os
# import numpy as np
# import glob

# def rank_grasps_with_uncertainty(npz_folder, gripper_size=(0.08, 0.02, 0.02), Wu=1e5):
#     """
#     Args:
#         npz_folder: folder containing .npz files with keys ['out2', 'std_out2', 'grasps', 'scores']
#         gripper_size: dimensions of gripper bounding box (width, depth, height)
#         Wu: weight for uncertainty penalty
#     Returns:
#         list of ranked grasps with updated scores
#     """
#     all_files = sorted(glob.glob(os.path.join(npz_folder, '*.npz')))
#     ranked_outputs = []

#     for fpath in all_files:
#         data = np.load(fpath)
#         out2 = data['out2']        # shape (N, 3), completed PCD
#         std_out2 = data['std_out2']  # shape (N, 3), standard deviation
#         grasps = data['grasps']    # shape (M, 4, 4), grasp poses as 4x4 matrices
#         scores = data['scores']    # shape (M,), original scores

#         std_norm = np.linalg.norm(std_out2, axis=1)  # (N,)

#         updated_scores = []
#         for i, grasp in enumerate(grasps):
#             # Crop points within gripper bounding box
#             grasp_frame = grasp[:3, :3]  # Rotation
#             grasp_pos = grasp[:3, 3]     # Translation

#             # Transform point cloud to grasp frame
#             pcd_local = (out2 - grasp_pos) @ grasp_frame  # (N, 3)

#             # Check which points fall inside gripper bounding box
#             half_w, half_d, half_h = np.array(gripper_size) / 2
#             mask = (
#                 (np.abs(pcd_local[:, 0]) <= half_w) &
#                 (np.abs(pcd_local[:, 1]) <= half_d) &
#                 (np.abs(pcd_local[:, 2]) <= half_h)
#             )

#             if np.sum(mask) == 0:
#                 # No points in grasp region → heavy penalty
#                 uncertainty_penalty = 1e3
#             else:
#                 uncertainty_penalty = np.mean(std_norm[mask])  # avg uncertainty

#             adjusted_score = scores[i] - Wu * uncertainty_penalty
#             updated_scores.append(adjusted_score)

#         updated_scores = np.array(updated_scores)
#         ranked_indices = np.argsort(updated_scores)[::-1]  # descending
#         top5 = ranked_indices[:5]

#         ranked_outputs.append({
#             'file': fpath,
#             'top5_indices': top5,
#             'top5_scores': updated_scores[top5],
#             'original_scores': scores[top5],
#             'mean_uncertainty': np.mean(std_norm)
#         })

#     # Sort files by lowest uncertainty
#     ranked_outputs.sort(key=lambda x: x['mean_uncertainty'])
#     return ranked_outputs


# # ==== Run it ====
# folder = 'log/PointAttN_baseline_cd_matching_f1_MC_cd_debug_pcn/all_T60_0.1'
# ranked_grasps = rank_grasps_with_uncertainty(folder)

# print("\n🍓 Ranked Predictions by Certainty (lower = more confident):")
# for i, entry in enumerate(ranked_grasps):
#     f = os.path.basename(entry['file'])
#     mean_std = entry['mean_uncertainty']
#     print(f"{i+1:2d}. {f} - Mean STD: {mean_std:.6f}")
#     for idx, score in zip(entry['top5_indices'], entry['top5_scores']):
#         print(f"     Grasp {idx:2d} - Adjusted Score: {score:.2f}")


# grasp_uncertainty.py
# Usage:
#   conda create -n grasp python=3.10 -y && conda activate grasp
#   pip install open3d numpy scipy
#   python grasp_uncertainty.py --cloud strawberry.ply --sigma sigma.npy --centroid 0.0 0.0 0.0


import argparse, numpy as np, open3d as o3d
from scipy.spatial.transform import Rotation as Rsc

def load_inputs(cloud_path, sigma_path, centroid_xyz):
    pcd = o3d.io.read_point_cloud(cloud_path)
    P = np.asarray(pcd.points, dtype=np.float64)
    sigma = np.load(sigma_path).astype(np.float64)
    assert len(P) == len(sigma), "Point cloud and sigma must have same length/order"
    c = np.array(centroid_xyz, dtype=np.float64)
    return pcd, P, sigma, c

def clean_and_normals(pcd, P, c, knn=40):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    N = np.asarray(pcd.normals, dtype=np.float64)
    # orient outward
    V = P - c
    flip = (np.einsum('ij,ij->i', N, V) < 0)
    N[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(N)
    return N

def curvature_and_pca(P, tree, idx, k=40):
    # local PCA around point idx
    _, nn_idx, _ = tree.search_knn_vector_3d(P[idx], k)
    Q = P[nn_idx]
    C = np.cov(Q.T)
    w, U = np.linalg.eigh(C)  # ascending eigenvalues
    # sort descending
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    curv = w[-1] / max(1e-12, w.sum())
    # principal dir = U[:,0]
    return curv, U

def sample_pad_points(center, t1, t2, half_w, half_h, depth=0.003, nx=6, ny=4):
    xs = np.linspace(-half_w, half_w, nx)
    ys = np.linspace(-half_h, half_h, ny)
    pts = []
    for x in xs:
        for y in ys:
            # thin slab across ±depth along t2 (normal of pad plane)
            pts.append(center + x*t1 + y*t2 + 0.0*t2)  # center surface
    return np.array(pts)

def mean_sigma_at(pts, tree, sigma):
    out = np.empty(len(pts), dtype=np.float64)
    for i, q in enumerate(pts):
        _, idx, _ = tree.search_knn_vector_3d(q, 1)
        out[i] = sigma[idx[0]]
    return float(out.mean())

def count_inliers_pad(pts, P, tree, pad_thick=0.004):
    # count scene points close to pad plane (approx by nearest neighbor distance threshold)
    cnt = 0
    for q in pts:
        _, idx, d2 = tree.search_knn_vector_3d(q, 1)
        if d2[0] <= pad_thick**2:
            cnt += 1
    return cnt

def tube_points(origin, a, length=0.06, radius=0.006, L=24):
    # sample a few rings along approach
    # build perp basis
    a = a / np.linalg.norm(a)
    tmp = np.array([1,0,0], dtype=np.float64)
    if abs(np.dot(a, tmp)) > 0.9: tmp = np.array([0,1,0], dtype=np.float64)
    x = np.cross(a, tmp); x /= np.linalg.norm(x)
    y = np.cross(a, x)
    pts=[]
    for s in np.linspace(0, length, L):
        for ang in np.linspace(0, 2*np.pi, 6, endpoint=False):
            pts.append(origin + s*a + radius*np.cos(ang)*x + radius*np.sin(ang)*y)
    return np.array(pts)

def load_from_npz(npz_path):
    data = np.load(npz_path)
    mean_out2 = data["mean_out2"]            # (N,3)
    std_xyz = data["std_out2"]     # (N, 3)



    # Option A: RMS across xyz
    sigma = np.sqrt((std_xyz**2).mean(axis=1))   # (N,)

    # Option B: max across xyz
    # sigma = std_xyz.max(axis=1)



    # save files for grasp_uncertainty.py
    np.save("strawberry_mean.npy", mean_out2)   # completed cloud
    np.save("strawberry_std.npy", sigma)        # scalar uncertainty

    print("Saved strawberry_mean.npy and strawberry_std.npy")


    # compute centroid automatically
    centroid = mean_out2.mean(axis=0)

    return mean_out2, sigma, centroid

def main(args):
    if args.npz is not None:
        P, sigma, c = load_from_npz(args.npz)
        # build an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P)
    else:
        # original path: read ply + sigma.npy + centroid args
        pcd, P, sigma, c = load_inputs(args.cloud, args.sigma, args.centroid)

    
    # Bounding box dimensions
    # mins = P.min(0)
    # maxs = P.max(0)
    # extent = maxs - mins   # [dx, dy, dz] in meters
    '''
    Strawberry extents (x,y,z): [0.21188802 0.04015682 0.03429589]
    Width  (x span): 211.9 mm
    Height (y span): 40.2 mm
    Length (z span): 34.3 mm
    Approx. fruit diameter: 211.9 mm
    '''


    # Robust extents (5-95%), remove outliers
    q_min = np.percentile(P, 5, axis=0)
    q_max = np.percentile(P, 95, axis=0)
    extent = q_max - q_min
    print("Robust extents (5-95%):", extent)
    '''
    Strawberry extents (x,y,z): [0.03204433 0.03465536 0.03047521]
    Width  (x span): 32.0 mm
    Height (y span): 34.7 mm
    Length (z span): 30.5 mm
    Approx. fruit diameter: 34.7 mm
    '''


    print(f"Strawberry extents (x,y,z): {extent}")
    print(f"  Width  (x span): {extent[0]*1000:.1f} mm")
    print(f"  Height (y span): {extent[1]*1000:.1f} mm")
    print(f"  Length (z span): {extent[2]*1000:.1f} mm")

    # Approximate "diameter" as the max of extents
    dia = extent.max()
    print(f"Approx. fruit diameter: {dia*1000:.1f} mm")

    # override width search
    w_min = 0.6 * dia
    w_max = 1.2 * dia

    print(f"[auto-width] fruit dia≈{dia*1000:.1f} mm -> w_min={w_min*1000:.1f} mm, w_max={w_max*1000:.1f} mm")



    N = clean_and_normals(pcd, P, c, knn=args.knn_normals)
    tree = o3d.geometry.KDTreeFlann(pcd)

    # certainty mask
    thr = np.quantile(sigma, args.certile)
    mask_cert = sigma <= thr

    # fruit long axis from PCA
    V = P - c
    C = np.cov(V.T)
    w, U = np.linalg.eigh(C)
    axis = U[:, np.argmax(w)]
    axis /= np.linalg.norm(axis)
    # sideways = nearly perpendicular to fruit axis
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    cosine = np.abs(Vn @ axis)
    side_mask = cosine < 0.3   # 0.3 ≈ >72° from axis

    # distance from centroid
    d = np.linalg.norm(P - c, axis=1)
    # keep only outer 10% farthest points (boundary)
    boundary_mask = d >= np.quantile(d, 0.9)

    # combine all three filters: certainty + sideways + boundary
    mask = mask_cert & side_mask & boundary_mask
    cand_idx = np.where(mask)[0]
    print("Candidates after certainty+side+boundary filter:", len(cand_idx))



    # score best grasp
    best = None
    lam, alpha, gamma = args.lam, args.alpha, args.gamma
    for idx in cand_idx:
        curv, U = curvature_and_pca(P, tree, idx, k=args.knn_pca)
        if curv > args.max_curv: 
            continue
        p = P[idx]
        # n = N[idx]
        # a = -n / (np.linalg.norm(n)+1e-12)  # approach inward
        # # build tangent basis: t1 = principal direction projected on tangent plane, t2 = a x t1
        # t1 = U[:,0]
        # # ensure t1 orth to a
        # t1 = t1 - (t1@a)*a
        # if np.linalg.norm(t1) < 1e-6:
        #     continue
        # t1 /= np.linalg.norm(t1)
        # t2 = np.cross(a, t1); t2 /= (np.linalg.norm(t2)+1e-12)


        # Use direction from centroid to point, but project to be perpendicular to fruit axis
        v = p - c
        a = v - (v @ axis) * axis   # remove component along axis
        if np.linalg.norm(a) < 1e-6:
            continue
        a /= np.linalg.norm(a)
        a = -a   # point inward toward centroid
        t1 = axis  # align jaw axis with fruit axis
        t2 = np.cross(a, t1)
        t2 /= (np.linalg.norm(t2)+1e-12)



        for w in np.linspace(w_min, w_max, args.w_steps):
            half_w = 0.5*w
            half_h = args.pad_half_h
            # pad centers
            cl = p - half_w*t1
            cr = p + half_w*t1
            # sample pad points
            Pl = sample_pad_points(cl, t1, t2, half_w=0.002, half_h=half_h)
            Pr = sample_pad_points(cr, t1, t2, half_w=0.002, half_h=half_h)

            # counts and sigma
            inl_l = count_inliers_pad(Pl, P, tree, pad_thick=args.pad_thickness)
            inl_r = count_inliers_pad(Pr, P, tree, pad_thick=args.pad_thickness)
            if min(inl_l, inl_r) < args.min_inliers: 
                continue

            msl = mean_sigma_at(Pl, tree, sigma)
            msr = mean_sigma_at(Pr, tree, sigma)
            # approach tube (start a bit outside)
            tube = tube_points(p + args.start_offset*a, a, length=args.approach_len, radius=args.tube_r, L=24)
            # mean sigma along approach (discourage uncertain approach)
            ms_tube = mean_sigma_at(tube, tree, sigma)

            score = (inl_l + inl_r) - lam*((msl+msr)/2.0 + alpha*ms_tube) - gamma*abs(inl_l - inl_r)

            if (best is None) or (score > best["score"]):
                best = dict(score=score, p=p, a=a, t1=t1, t2=t2, w=w, idx=idx,
                            inl_l=inl_l, inl_r=inl_r, msl=msl, msr=msr, ms_tube=ms_tube)

    if best is None:
        print("No feasible grasp found. Try relaxing thresholds (increase certile, max_curv, lower min_inliers).")
        return

    # Build pose
    R = np.stack([best["t1"], best["t2"], best["a"]], axis=1)  # columns
    # t = best["p"] - args.pose_offset*best["a"]
    t = best["p"] - args.pre_close_gap * best["a"]

    # Print results
    euler = Rsc.from_matrix(R).as_euler('xyz', degrees=True)
    print("\n=== UNCERTAINTY-AWARE GRASP ===")
    print(f"Score: {best['score']:.2f}")
    print(f"Translation t (m): {t}")
    print(f"Rotation R (rows):\n{R}")
    print(f"Euler XYZ (deg): {euler}")
    print(f"Jaw width w (m): {best['w']:.3f}")
    print(f"Inliers L/R: {best['inl_l']}/{best['inl_r']}  meanσ pads: {(best['msl']+best['msr'])/2:.4f}  meanσ tube: {best['ms_tube']:.4f}")

    # Optional: save as JSON for your controller
    if args.save_json:
        import json
        out = dict(t=t.tolist(), R=R.tolist(), width=float(best["w"]), score=float(best["score"]))
        with open(args.save_json, "w") as f: json.dump(out, f, indent=2)
        print(f"Saved grasp to {args.save_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud")
    ap.add_argument("--sigma")
    ap.add_argument("--centroid", nargs=3, type=float)

    ap.add_argument("--lam", type=float, default=0.5,
                    help="Weight for uncertainty penalty at contacts")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Relative weight for uncertainty along approach path")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="Penalty for imbalance between left/right contacts")

    # geometry / thresholds
    ap.add_argument("--certile", type=float, default=0.2)      # keep lowest 20% σ, increase if you want more candidates
    ap.add_argument("--max_curv", type=float, default=0.2)
    ap.add_argument("--knn_normals", type=int, default=40)
    ap.add_argument("--knn_pca", type=int, default=40)
    ap.add_argument("--w_steps", type=int, default=8)
    ap.add_argument("--pad_half_h", type=float, default=0.007)
    ap.add_argument("--pad_thickness", type=float, default=0.004)
    ap.add_argument("--min_inliers", type=int, default=20)
    ap.add_argument("--approach_len", type=float, default=0.06)
    ap.add_argument("--tube_r", type=float, default=0.006)
    ap.add_argument("--start_offset", type=float, default=0.02)
    ap.add_argument("--pose_offset", type=float, default=0.015)
    ap.add_argument("--save_json", default="grasp_pose.json")
    ap.add_argument("--npz", help="Path to .npz file containing mean_out2 and std_out2")
    ap.add_argument("--pre_close_gap", type=float, default=0.003,
                help="Visual/command gap before closing (m). Smaller = closer inside the strawberry")
    args = ap.parse_args()
    main(args)

'''
conda activate pointAttn


cloud: most-certain strawberry completion pcl
sigma: per point stddev (check mean_out2 field)
centroid: strawberry centroid

# python grasping.py \
#   --cloud strawberry.ply \
#   --sigma sigma.npy \
#   --centroid 0.12 0.03 0.40 \
#   --save_json grasp_pose.json


python grasping.py \
  --npz log/PointAttN_baseline_cd_matching_f1_cd_debug_pcn/all/batch2_sample12_data.npz \
  --certile 0.6 \
  --max_curv 0.5 \
  --min_inliers 3 \
  --approach_len 0.04 --tube_r 0.004

python grasping.py \
  --npz log/PointAttN_baseline_cd_matching_f1_MC_cd_debug_pcn/all_T60_0.1/batch1_sample13_data.npz \
  --certile 0.6 \
  --max_curv 0.5 \
  --min_inliers 3 \
  --approach_len 0.04 --tube_r 0.004


output: translation and rotation of the grasp, and jaw width w
{
  "t": [x,y,z],
  "R": [[...],[...],[...]],
  "width": 0.034,
  "score": 128.7
}
Feed this to your robot: move to t + approach_len * a, descend along -a to t, close to width, retreat.
'''