def render_avatar(gaussian_data, camera_data, save_path=None):
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from scipy.spatial.transform import Rotation as R
    import cv2
    import os
    import torch

    # Datos originales en World Space.
    means3D = gaussian_data["means3D"]
    rotations_quat = gaussian_data["rotations"]
    scales = gaussian_data["scales"]
    opacities = gaussian_data["opacities"]
    colors = gaussian_data["shs"].squeeze(1)

    original_view_matrix = camera_data["view_matrix"].to(torch.device("cuda"))

    # --- INICIO DE LA LÓGICA DE PRE-TRANSFORMACIÓN CORRECTA ---

    # 1. Separamos la matriz de vista en su parte de transformación 3x3 (M) y traslación (T).
    M = original_view_matrix[:3, :3]
    T = original_view_matrix[:3, 3]

    # 2. Transformamos las POSICIONES. Esta parte ya funcionaba y es lo que evita la pantalla blanca.
    #    p_cam = p_world @ M.T + T
    means3D_transformed = means3D @ M.T + T

    # 3. Transformamos las ROTACIONES. Aquí está la corrección crucial.
    #    No podemos usar M directamente porque contiene una reflexión (det(M) < 0).
    #    Usamos SVD para extraer la rotación pura de M. M = U @ S @ V.T
    #    La rotación pura más cercana es R = U @ V.T
    U, _, V_t = torch.linalg.svd(M)
    R_pure = U @ V_t

    # Si el determinante es negativo, la matriz R_pure todavía contiene la reflexión.
    # Debemos eliminarla para obtener una rotación verdadera (determinante = +1).
    # Esto es esencial para no "voltear" los gaussianos.
    if torch.det(R_pure) < 0:
        U_clone = U.clone()
        U_clone[:, -1] *= -1  # Invertimos el último vector singular
        R_pure = U_clone @ V_t

    # Ahora R_pure es una matriz de rotación garantizada.
    # Convertimos las rotaciones originales de los gaussianos a matrices.
    rot_matrices_orig = torch.from_numpy(
        R.from_quat(rotations_quat.cpu().numpy()).as_matrix()
    ).to(torch.device("cuda"), dtype=torch.float32)

    # Componemos las rotaciones: R_transformada = R_vista_pura @ R_original
    # Esto es matemáticamente correcto porque ambas son rotaciones puras.
    rot_matrices_transformed = R_pure @ rot_matrices_orig

    # Convertimos el resultado de vuelta a cuaterniones para el rasterizador.
    rotations_transformed = torch.from_numpy(
        R.from_matrix(rot_matrices_transformed.cpu().numpy()).as_quat()
    ).to(torch.device("cuda"), dtype=torch.float32)

    # --- FIN DE LA LÓGICA ---

    # Como hemos pre-transformado todo al espacio de la cámara,
    # le decimos al rasterizador que la cámara está en el origen (identidad).
    canonical_view_matrix = torch.eye(4, device=torch.device("cuda"))
    canonical_camera_pos = torch.zeros(3, device=torch.device("cuda"))

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera_data["height"]),
        image_width=int(camera_data["width"]),
        tanfovx=camera_data["tanfovx"],
        tanfovy=camera_data["tanfovy"],
        bg=torch.ones(3, dtype=torch.float32, device=torch.device("cuda")),
        scale_modifier=1.0,
        viewmatrix=canonical_view_matrix,
        projmatrix=camera_data["proj_matrix"].to(torch.device("cuda")),
        sh_degree=3,
        campos=canonical_camera_pos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings)

    # Renderizamos con los datos pre-transformados
    rendered_image, _, _, _ = rasterizer(
        means3D=means3D_transformed,
        means2D=torch.zeros_like(means3D_transformed),
        opacities=opacities,
        shs=None,
        colors_precomp=colors,
        scales=scales,
        rotations=rotations_transformed,
    )

    if save_path is not None:
        img_to_save = (rendered_image.clamp(0.0, 1.0) * 255).to(torch.uint8).cpu().numpy()
        img_to_save = img_to_save.transpose(1, 2, 0)
        img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_to_save)

        ply_path = save_path.replace(".png", ".ply")
        try:
            import open3d as o3d
            point_cloud = o3d.geometry.PointCloud()
            # Los puntos ya están en el espacio de la cámara, listos para guardar.
            point_cloud.points = o3d.utility.Vector3dVector(means3D_transformed.cpu().numpy())
            point_cloud.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
            o3d.io.write_point_cloud(ply_path, point_cloud)
            print(f"Saved rendered image to {save_path} and point cloud to {ply_path}")
        except ImportError:
            print(f"Saved rendered image to {save_path}. Open3D not installed, skipping point cloud save.")
            
    return rendered_image


gaussian_data = {
    "means3D":(gs_model_list[0].offset_xyz + query_points.squeeze(1)).squeeze(0),
    "opacities": gs_model_list[0].opacity,
    "scales": gs_model_list[0].scaling,
    "rotations": gs_model_list[0].rotation,
    "shs": gs_model_list[0].shs
}
import torch

device = torch.device("cuda")

import torch

device = torch.device("cuda")

camera_data = {
    "camera_position": torch.tensor([-0.0512, -0.6303, 2.4159], device=device),
    "view_matrix": torch.tensor([
        [ 0.9996,  0.0256, -0.0092,  0.0895],
        [ 0.0259, -0.9991,  0.0344, -0.6492],
        [-0.0083, -0.0346, -0.9994,  2.3676],
        [ 0.0,     0.0,     0.0,     1.0]
    ], device=device),
    "proj_matrix": torch.tensor([
        [ 2.3094,  0.0,     0.0,     0.0],
        [ 0.0,     1.7321,  0.0,     0.0],
        [ 0.0,     0.0,     1.0020,  1.0],
        [ 0.0,     0.0,    -0.2002,  0.0]
    ], device=device),
    "tanfovx": torch.tensor(0.4330127018922193, device=device),
    "tanfovy": torch.tensor(0.5773502691896257, device=device),
    "height": torch.tensor(1920, device=device),
    "width": torch.tensor(1440, device=device)
}

render_avatar(gaussian_data, camera_data, save_path="/data1/users/adrian/LHM/exps/images/sweater2.png")


smplx_data["body_pose"] = torch.zeros_like(smplx_data["body_pose"])
smplx_data["trans"] = torch.zeros_like(smplx_data["trans"])
smplx_data["root_pose"] = torch.zeros_like(smplx_data["root_pose"])