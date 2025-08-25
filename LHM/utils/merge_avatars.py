import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
        
from LHM.models.rendering.gs_renderer import GS3DRenderer, PointEmbed

def get_gaussian_visibility(gaussians_data: dict, camera_data: dict) -> torch.Tensor:
    """
    Calculates a visibility score for each gaussian based on its contribution to the rendered image.

    The function renders the scene and then
    computes the gradient of the total image brightness with respect to the opacity of each 
    gaussian. A gaussian with a high gradient is highly visible and influential in the scene, 
    while a gaussian with a zero gradient is likely occluded or outside the camera's view.

    Parameters:
    - gaussians_data: dictionary, containing gaussian properties for a single avatar:
        - means3D: torch.Tensor, shape (B, 3)          # 3D positions 
        - opacities: torch.Tensor, shape (B, 1)        # Opacity values
        - shs: torch.Tensor, shape (B, K, 3)           # Spherical harmonics coefficients 
        - scales: torch.Tensor, shape (B, 3)           # Scale factors 
        - rotations: torch.Tensor, shape (B, 4)        # Rotation quaternions

    - camera_data: dictionary, containing parameters for a single camera:
        - camera_position: torch.Tensor, shape (3,)    # Camera position in world coordinates
        - view_matrix: torch.Tensor, shape (4, 4)      # Camera view matrix
        - proj_matrix: torch.Tensor, shape (4, 4)      # Camera projection matrix
        - tanfovx: float                               # Tangent of horizontal field of view
        - tanfovy: float                               # Tangent of vertical field of view
        - height: int                                  # Image height in pixels
        - width: int                                   # Image width in pixels

    Returns:
    - visibility_scores: torch.Tensor, shape (B,)      # A score [0, 1] for each gaussian indicating its visibility.
    """
    # ---- Setup gradients and rasterizer ----
    opacities_for_grad = gaussians_data["opacities"].clone().detach().requires_grad_(True)
    
    shs_for_grad = gaussians_data["shs"].clone().detach().requires_grad_(True)
    scales_for_grad = gaussians_data["scales"].clone().detach().requires_grad_(True)
    rotations_for_grad = gaussians_data["rotations"].clone().detach().requires_grad_(True)
    means3D = gaussians_data["means3D"].clone().detach()

    original_view_matrix = camera_data["view_matrix"].to(torch.device("cuda"))
    R_view = original_view_matrix[:3, :3]
    T_view = original_view_matrix[:3, 3]
    canonical_view_matrix = torch.eye(4).to(torch.device("cuda"))
    canonical_camera_pos = torch.zeros(3).to(torch.device("cuda"))
    means3D_transformed = means3D @ R_view.T + T_view
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera_data["height"]),
        image_width=int(camera_data["width"]),
        tanfovx=camera_data["tanfovx"],
        tanfovy=camera_data["tanfovy"],
        bg=torch.zeros(3, device=opacities_for_grad.device),
        scale_modifier=1.0,
        viewmatrix=canonical_view_matrix,
        projmatrix=camera_data["proj_matrix"].to(torch.device("cuda")),
        sh_degree=shs_for_grad.shape[-2] - 1,
        campos=canonical_camera_pos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ---- Forward pass: render the image using the tensors marked for grads ----
    rendered_image, _, _, _ = rasterizer(
        means3D=means3D_transformed,
        means2D=torch.zeros_like(means3D_transformed, requires_grad=True, device=means3D_transformed.device, dtype=means3D_transformed.dtype),
        opacities=opacities_for_grad,
        shs=shs_for_grad,             # Use the new tensor
        scales=scales_for_grad,       # Use the new tensor
        rotations=rotations_for_grad  # Use the new tensor
    )

    # ---- Backward pass: compute visibility from gradients ----
    rendered_image.sum().backward()
    
    with torch.no_grad():
        # The gradient of a gaussian's opacity represents its contribution/visibility.
        opacity_gradients = opacities_for_grad.grad
            
        # Use the absolute value of the gradient and remove singleton dimensions (B, 1) -> (B,).
        visibility_scores = opacity_gradients.abs().squeeze()
        max_score = visibility_scores.max()
        
        # Normalize scores to a [0, 1] range for easier use as a mask or weight.
        if max_score > 1e-9:
            visibility_scores /= max_score
            
    return visibility_scores



def merge_avatars(gaussians_data, cameras_data):
    """
    Merges multiple avatars with B gaussians each into a single image based on their positions, colors, opacities, scales, and rotations.
    The merging is done by computing the visibility of each avatar's gaussians from the camera's perspective and then averaging their contributions.
    For the positions, the ponderation is done per axis. The more parallel is an axis to the camera-gaussian vector, the less it contributes to the final position,
    while the more perpendicular it is, the more it contributes.

    Parameters:
    - gaussians_data: list of dictionaries, each containing:
        - means3D: torch.Tensor, shape (B, 3)          # 3D positions 
        - opacities: torch.Tensor, shape (B,1)         # Opacity values
        - shs: torch.Tensor, shape (B, 1, 3)              # Spherical harmonics coefficients 
        - scales: torch.Tensor, shape (B, 3)           # Scale factors 
        - rotations: torch.Tensor, shape (B, 4)        # Rotation quaternions

    - cameras_data: list of dictionaries, each containing camera parameters:
        - camera_position: torch.Tensor, shape (3,)    # Camera position in world coordinates
        - view_matrix: torch.Tensor, shape (4, 4)      # Camera view matrix
        - proj_matrix: torch.Tensor, shape (4, 4)      # Camera projection matrix
        - tanfovx: float                               # Tangent of horizontal field of view
        - tanfovy: float                               # Tangent of vertical field of view
        - height: int                                  # Image height in pixels
        - width: int                                   # Image width in pixels

    Returns:
    - output: dictionary containing merged avatar data:
        - means3D: torch.Tensor, shape (B, 3)          # Merged 3D positions 
        - opacities: torch.Tensor, shape (B, 1)        # Merged opacity values
        - shs: torch.Tensor, shape (B, 3)              # Merged spherical harmonics coefficients 
        - scales: torch.Tensor, shape (B, 3)           # Merged scale factors 
        - rotations: torch.Tensor, shape (B, 4)        # Merged rotation quaternions
    """

    # ---- Compute visibility for each avatar ----
    gaussian_vis_list = []
    for gaussians, camera in zip(gaussians_data, cameras_data):
        visible_mask = get_gaussian_visibility(
            gaussians_data=gaussians,
            camera_data=camera
        )
        gaussian_vis_list.append(visible_mask)
    gaussian_vis = torch.stack(gaussian_vis_list, dim=0).unsqueeze(-1) + 1e-8  # (N, B, 1)
    # (N, B, 1) -> (B, 1)
    total_gaussian_vis = torch.sum(gaussian_vis, dim=0)

    # ---- Stack data ----
    camera_positions = torch.stack([camera["camera_position"] for camera in cameras_data], dim=0).to(gaussians_data[0]["means3D"].device) # (N, 3)
    avatar_positions = torch.stack([gaussian["means3D"] for gaussian in gaussians_data], dim=0)  # (N, B, 3)
    gaussian_colors = torch.stack([gaussian["shs"].squeeze(1) for gaussian in gaussians_data], dim=0)  # (N, B, 3)
    gaussian_opacities = torch.stack([gaussian["opacities"] for gaussian in gaussians_data], dim=0)  # (N, B, 1)
    gaussian_scales = torch.stack([gaussian["scales"] for gaussian in gaussians_data], dim=0)  # (N, B, 3)
    gaussian_rotations = torch.stack([gaussian["rotations"] for gaussian in gaussians_data], dim=0)  # (N, B, 3, 3)

    # ---- Compute merged positions ----
    # (N, B, 3) - (N, 1, 3) -> (N, B, 3)
    view_vectors = avatar_positions - camera_positions.unsqueeze(1)
    
    # (N, B, 3) / (N, B, 1) -> (N, B, 3)
    view_vectors_norm = view_vectors / torch.norm(view_vectors, dim=2, keepdim=True)

    # (N, B, 3) * (N, B, 1) -> (N, B, 3)
    position_ponderations = torch.acos(view_vectors_norm.abs()) * gaussian_vis  + 1e-8
    
    # (N, B, 3) * (N, B, 3) -> (N, B, 3)
    weighted_positions = avatar_positions * position_ponderations
    
    # (N, B, 3) -> (B, 3)
    sum_weighted_positions = torch.sum(weighted_positions, dim=0)
    sum_ponderations = torch.sum(position_ponderations, dim=0)
    
    # (B, 3) / (B, 1) -> (B, 3)
    final_positions = sum_weighted_positions / sum_ponderations

    # ---- Compute merged colors ----
    # (N, B, 3) * (N, B, 1) -> (N, B, 3)
    weighted_colors = gaussian_colors * gaussian_vis

    # (N, B, 3) -> (B, 3)
    sum_weighted_colors = torch.sum(weighted_colors, dim=0)

    # (B, 3) / (B, 1) -> (B, 3)
    final_colors = sum_weighted_colors / total_gaussian_vis

    # ---- Compute merged opacities ----
    # (N, B, 1) * (N, B, 1) -> (N, B, 1)
    weighted_opacities = gaussian_opacities * gaussian_vis
    # (N, B, 1) -> (B, 1)
    sum_weighted_opacities = torch.sum(weighted_opacities, dim=0)
    # (B, 1) / (B, 1) -> (B, 1)
    final_opacities = sum_weighted_opacities / total_gaussian_vis

    # ---- Compute scale and rotation ----
    N, B, _ = gaussian_rotations.shape
    # general view ponderation, prioritizes first and second images, and penalizes the rest
    general_ponderation = torch.tensor([1.0, 1.0] + [0.1]*(N-2), device=gaussian_vis.device).view(N, 1, 1)  # (N, 1, 1)
    scale_rotation_ponderation = gaussian_vis * general_ponderation  # (N, B, 1)
    max_vis_indices = torch.argmax(scale_rotation_ponderation.squeeze(-1), dim=0)
    
    # (B,) -> (1, B, 1) -> (1, B, 4)
    idx_rot = max_vis_indices.view(1, B, 1).expand(1, B, 4)
    
    # (B,) -> (1, B, 1) -> (1, B, 3)
    idx_scale = max_vis_indices.view(1, B, 1).expand(1, B, 3)
    
    # (1, B, 4) -> (B, 4)
    final_rotations = torch.gather(gaussian_rotations, 0, idx_rot).squeeze(0)
    
    # El resultado se aprieta de (1, B, 3) a (B, 3).
    final_scales = torch.gather(gaussian_scales, 0, idx_scale).squeeze(0)

    output = {
        "means3D": final_positions,
        "opacities": final_opacities,
        "shs": final_colors,
        "scales": final_scales,
        "rotations": final_rotations,
    }


    return output