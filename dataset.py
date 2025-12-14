# import os
# import torch
# import trimesh
# import numpy as np
# from torch.utils.data import Dataset

# class VoxelDataset(Dataset):
#     def __init__(self, obj_dir, grid=24):
#         self.files = [
#             os.path.join(obj_dir, f)
#             for f in os.listdir(obj_dir)
#             if f.endswith(".obj")
#         ]
#         self.grid = grid

#     def __len__(self):
#         return len(self.files) * 20  # 🔥 fake diversity

#     def __getitem__(self, idx):
#         mesh = trimesh.load(self.files[0], force="mesh")

#         mesh.vertices -= mesh.bounding_box.centroid
#         mesh.vertices /= np.max(mesh.bounding_box.extents)

#         pitch = 2.0 / self.grid
#         vox = mesh.voxelized(pitch=pitch)
#         mat = vox.matrix.astype(np.float32)

#         padded = np.zeros((self.grid, self.grid, self.grid), dtype=np.float32)
#         sx, sy, sz = mat.shape
#         padded[:sx, :sy, :sz] = mat

#         # 🔥 noise augmentation
#         padded += np.random.normal(0, 0.02, padded.shape)
#         padded = np.clip(padded, 0, 1)

#         return torch.tensor(padded).unsqueeze(0)
import os
import torch
import trimesh
import numpy as np
from torch.utils.data import Dataset

class VoxelDataset(Dataset):
    def __init__(self, obj_dir, grid=24):
        self.files = [
            os.path.join(obj_dir, f)
            for f in os.listdir(obj_dir)
            if f.endswith(".obj")
        ]
        self.grid = grid

    def __len__(self):
        return len(self.files) * 10

    def __getitem__(self, idx):
        mesh = trimesh.load(self.files[0], force="mesh")

        mesh.vertices -= mesh.bounding_box.centroid
        mesh.vertices /= np.max(mesh.bounding_box.extents)

        pitch = 2.0 / self.grid
        vox = mesh.voxelized(pitch=pitch)
        mat = vox.matrix.astype(np.float32)

        padded = np.zeros((self.grid, self.grid, self.grid), dtype=np.float32)
        sx, sy, sz = mat.shape
        padded[:sx, :sy, :sz] = mat

        padded += np.random.normal(0, 0.02, padded.shape)
        padded = np.clip(padded, 0, 1)

        return torch.tensor(padded).unsqueeze(0)
