# import torch
# import numpy as np
# import trimesh
# from skimage import measure
# from gan import Generator

# Z_DIM = 128

# G = Generator(Z_DIM)
# G.load_state_dict(torch.load("generator.pth", map_location="cpu"))
# G.eval()

# with torch.no_grad():
#     z = torch.randn(1, Z_DIM)
#     vox = G(z).squeeze().numpy()

# vox = (vox > 0).astype(np.float32)

# verts, faces, _, _ = measure.marching_cubes(vox, level=0.5)

# mesh = trimesh.Trimesh(verts, faces)
# mesh.export("generated_car.obj")
# mesh.show()
import torch
import numpy as np
import trimesh
from skimage import measure
from gan import Generator

Z_DIM = 64

G = Generator(Z_DIM)
G.load_state_dict(torch.load("generator.pth", map_location="cpu"))
G.eval()

with torch.no_grad():
    z = torch.randn(1, Z_DIM)
    vox = G(z).squeeze().numpy()

vox = (vox > 0).astype(np.float32)

verts, faces, _, _ = measure.marching_cubes(vox, level=0.5)

mesh = trimesh.Trimesh(verts, faces)
mesh.export("generated_car.obj")
mesh.show()
