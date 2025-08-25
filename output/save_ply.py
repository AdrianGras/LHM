from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch 
# 1. Preparar datos y ruta de salida
output_path = Path("output/scene.ply")
num_gaussians = gaussians_data["means3D"].shape[0]

# 2. Extraer y procesar los tensores
means3D = gaussians_data["means3D"].detach().cpu().numpy()
opacities = gaussians_data["opacities"].detach().cpu().numpy()
scales = torch.exp(gaussians_data["scales"]).detach().cpu().numpy()
rotations = F.normalize(gaussians_data["rotations"], p=2, dim=1).detach().cpu().numpy()

# Asumimos que "shs" contiene el color base (componente DC), lo que es típico.
# Tiene forma (N, 1, 3), lo quitamos para tener (N, 3).
colors_dc = gaussians_data["shs"].squeeze(1).detach().cpu().numpy()

# El formato PLY para splatting requiere los otros coeficientes de SH (componentes AC).
# Como no los tenemos, los rellenamos con ceros. Para sh_degree=3, se necesitan 45.
shs_rest = np.zeros((num_gaussians, 45), dtype=np.float32)

# 3. Construir el array estructurado de NumPy para el formato PLY
dtype_full = [
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ('opacity', 'f4'),
    ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
    ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
]
dtype_full.extend([(f'f_rest_{i}', 'f4') for i in range(shs_rest.shape[1])])
ply_dtype = np.dtype(dtype_full)
ply_data = np.empty(num_gaussians, dtype=ply_dtype)

# 4. Rellenar el array con los datos procesados
ply_data['x'], ply_data['y'], ply_data['z'] = means3D.T
ply_data['nx'], ply_data['ny'], ply_data['nz'] = np.zeros_like(means3D).T
ply_data['f_dc_0'], ply_data['f_dc_1'], ply_data['f_dc_2'] = colors_dc.T
ply_data['opacity'] = opacities.flatten()
ply_data['scale_0'], ply_data['scale_1'], ply_data['scale_2'] = scales.T
ply_data['rot_0'], ply_data['rot_1'], ply_data['rot_2'], ply_data['rot_3'] = rotations.T
for i in range(shs_rest.shape[1]):
    ply_data[f'f_rest_{i}'] = shs_rest[:, i]

# 5. Escribir el archivo PLY
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    header = f"""ply
format ascii 1.0
element vertex {num_gaussians}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
"""
    for i in range(shs_rest.shape[1]):
        header += f"property float f_rest_{i}\n"
    header += "end_header\n"
    
    f.write(header)
    np.savetxt(f, ply_data, fmt='%f')

print(f"Archivo PLY guardado en: {output_path}")