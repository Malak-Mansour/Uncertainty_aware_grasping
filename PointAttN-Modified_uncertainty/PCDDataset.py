# import torch, os, numpy as np

# class PCDDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir, augment=False, aug_params=None):
#         self.data_files = [f for f in os.listdir(data_dir) if f.startswith('processed_dataset_')]
#         self.data_dir = data_dir
#         self.augment = augment
        
#         # Default augmentation parameters
#         self.aug_params = {
#             'noise_points': 8,
#             'noise_range': (-0.2, 0.2),
#             'random_rotate': True,
#             'random_jitter': True,
#             'jitter_sigma': 0.01,
#             'random_scale': (0.8, 1.2)
#         }
        
#         # Update with provided parameters
#         if aug_params:
#             self.aug_params.update(aug_params)
    
#     def __len__(self):
#         return len(self.data_files)
    
#     def augment_pointcloud(self, src_pcd, model_pcd_transformed):
#         """Apply augmentation to point clouds"""
#         src_pcd = src_pcd.copy()
#         model_pcd_transformed = model_pcd_transformed.copy()
        
#         # Add noise to random points
#         if self.aug_params['noise_points'] > 0:
#             num_points = len(src_pcd)
#             if num_points >= self.aug_params['noise_points']:
#                 random_indices = np.random.choice(num_points, self.aug_params['noise_points'], replace=False)
#                 noise_range = self.aug_params['noise_range']
#                 noise = np.random.uniform(noise_range[0], noise_range[1], 
#                                          (self.aug_params['noise_points'], 3))
#                 src_pcd[random_indices] += noise
        
#         # Random rotation around y-axis
#         if self.aug_params['random_rotate']:
#             angle = np.random.uniform(0, 2 * np.pi)
#             cos_theta, sin_theta = np.cos(angle), np.sin(angle)
#             rotation_matrix = np.array([
#                 [cos_theta, 0, sin_theta],
#                 [0, 1, 0],
#                 [-sin_theta, 0, cos_theta]
#             ])
#             src_pcd = np.dot(src_pcd, rotation_matrix)
#             model_pcd_transformed = np.dot(model_pcd_transformed, rotation_matrix)
        
#         # Add small jitter to all points
#         if self.aug_params['random_jitter']:
#             src_pcd += np.random.normal(0, self.aug_params['jitter_sigma'], size=src_pcd.shape)
        
#         # Random scaling
#         if self.aug_params['random_scale']:
#             scale = np.random.uniform(self.aug_params['random_scale'][0], 
#                                      self.aug_params['random_scale'][1])
#             src_pcd *= scale
#             model_pcd_transformed *= scale
        
#         return src_pcd, model_pcd_transformed
    
#     def __getitem__(self, idx):
#         file_path = os.path.join(self.data_dir, self.data_files[idx])
#         data = np.load(file_path)
        
#         src_pcd = data['src_pcd']
#         model_pcd_transformed = data['model_pcd_transformed']
        
#         # Apply augmentation if enabled
#         if self.augment:
#             src_pcd, model_pcd_transformed = self.augment_pointcloud(src_pcd, model_pcd_transformed)
          
        
#         return {
#             'src_pcd': torch.from_numpy(src_pcd).float(),
#             'model_pcd_transformed': torch.from_numpy(model_pcd_transformed).float(),
#         }



# If doing occlusions
import torch, os, numpy as np

class PCDDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, augment=False, aug_params=None, num_points=2048):
        self.data_files = [f for f in os.listdir(data_dir) if f.startswith('processed_dataset_') and f.endswith('.npz')]
        self.data_files.sort()
        self.data_dir = data_dir
        self.augment = augment
        self.num_points = int(num_points)

        # Default augmentation parameters
        self.aug_params = {
            'noise_points': 8,
            'noise_range': (-0.2, 0.2),
            'random_rotate': True,
            'random_jitter': True,
            'jitter_sigma': 0.01,
            'random_scale': (0.8, 1.2)
        }
        if aug_params:
            self.aug_params.update(aug_params)

    def __len__(self):
        return len(self.data_files)

    def _resample_points(self, pts):
        """Return pts with exactly self.num_points rows by sampling or repeating."""
        n = int(pts.shape[0])
        if n == 0:
            # Extremely occluded: synthesize a tiny cloud at origin to avoid crashes
            pts = np.zeros((1, 3), dtype=np.float32)
            n = 1
        if n >= self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
        else:
            idx = np.random.choice(n, self.num_points, replace=True)  # repeat some points
        return pts[idx]

    def augment_pointcloud(self, src_pcd, model_pcd_transformed):
        """Apply augmentation to point clouds (src affects model with same transform)."""
        src_pcd = src_pcd.copy()
        model_pcd_transformed = model_pcd_transformed.copy()

        # Add noise to random points (src only)
        if self.aug_params['noise_points'] > 0 and len(src_pcd) > 0:
            k = min(self.aug_params['noise_points'], len(src_pcd))
            idx = np.random.choice(len(src_pcd), k, replace=False)
            lo, hi = self.aug_params['noise_range']
            noise = np.random.uniform(lo, hi, (k, 3)).astype(src_pcd.dtype)
            src_pcd[idx] += noise

        # Random rotation around y-axis
        if self.aug_params['random_rotate']:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s],
                          [0, 1, 0],
                          [-s, 0, c]], dtype=src_pcd.dtype)
            if len(src_pcd) > 0:
                src_pcd = src_pcd @ R.T
            if len(model_pcd_transformed) > 0:
                model_pcd_transformed = model_pcd_transformed @ R.T

        # Small jitter (src only)
        if self.aug_params['random_jitter'] and len(src_pcd) > 0:
            src_pcd = src_pcd + np.random.normal(0, self.aug_params['jitter_sigma'], size=src_pcd.shape).astype(src_pcd.dtype)

        # Random scaling (both)
        if self.aug_params['random_scale']:
            lo, hi = self.aug_params['random_scale']
            scale = np.random.uniform(lo, hi)
            if len(src_pcd) > 0:
                src_pcd = (src_pcd * scale).astype(src_pcd.dtype)
            if len(model_pcd_transformed) > 0:
                model_pcd_transformed = (model_pcd_transformed * scale).astype(model_pcd_transformed.dtype)

        return src_pcd, model_pcd_transformed

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(file_path)

        # Load required fields
        src_pcd = data['src_pcd'].astype(np.float32)
        model_pcd_transformed = data['model_pcd_transformed'].astype(np.float32)

        # Optional augmentation
        if self.augment:
            src_pcd, model_pcd_transformed = self.augment_pointcloud(src_pcd, model_pcd_transformed)

        # Enforce fixed size for PointAttN
        src_pcd = self._resample_points(src_pcd)
        model_pcd_transformed = self._resample_points(model_pcd_transformed)

        return {
            'src_pcd': torch.from_numpy(src_pcd).float(),
            'model_pcd_transformed': torch.from_numpy(model_pcd_transformed).float(),
        }
