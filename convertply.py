python -c '
import torch
import numpy as np
from tinysplat.gaussian_model import GaussianModel

# 1. Initialize and load
gaussians = GaussianModel(sh_degree=3)
ckpt = torch.load("checkpoints/checkpoint_30000.pth", map_location="cpu", weights_only=False)
state_dict = ckpt.get("gaussians", ckpt.get("state_dict", ckpt))
gaussians.restore(state_dict, None)

# 2. Extract raw attributes as numpy arrays
xyz = gaussians.xyz.detach().numpy()
f_dc = gaussians.features_dc.detach().numpy().reshape(-1, 3)
f_rest = gaussians.features_rest.detach().numpy().reshape(-1, 45) if hasattr(gaussians, "features_rest") else np.zeros((len(xyz), 45))
opacity = gaussians.opacity.detach().numpy()
scale = gaussians.scales.detach().numpy()
rot = gaussians.rotations.detach().numpy()

num_vertex = len(xyz)

# 3. Construct standard 3DGS binary PLY format
with open("output_model_30000.ply", "wb") as f:
    # Write header
    f.write(b"ply\nformat binary_little_endian 1.0\n")
    f.write(f"element vertex {num_vertex}\n".encode())
    f.write(b"property float x\nproperty float y\nproperty float z\n")
    f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
    for i in range(3): f.write(f"property float f_dc_{i}\n".encode())
    for i in range(45): f.write(f"property float f_rest_{i}\n".encode())
    f.write(b"property float opacity\n")
    for i in range(3): f.write(f"property float scale_{i}\n".encode())
    for i in range(4): f.write(f"property float rot_{i}\n".encode())
    f.write(b"end_header\n")
    
    # Pack data matching standard layout (xyz, normals [0,0,0], features, opacity, scales, rotations)
    normals = np.zeros_like(xyz)
    data = np.hstack([xyz, normals, f_dc, f_rest, opacity, scale, rot]).astype(np.float32)
    f.write(data.tobytes())

print("Successfully generated output_model_30000.ply")