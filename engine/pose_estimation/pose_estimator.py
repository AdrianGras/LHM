# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Peihao Li
# @Email         : liphao99@gmail.com
# @Time          : 2025-03-11 12:47:58
# @Function      : inference code for pose estimation

import os
import sys

sys.path.append("./")

import pdb
from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from engine.ouputs import BaseOutput
from engine.pose_estimation.model import load_model
from scipy.spatial.transform import Rotation as ScipyRotation
from typing import Optional, Dict, Any

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


@dataclass
class SMPLXOutput:
    beta: Optional[np.ndarray]
    is_full_body: bool
    ratio: Optional[float]
    msg: str
    camera_data: Optional[Dict[str, Any]] = None  # Cámara en el "mundo real"
    smplx_params: Optional[Dict[str, Any]] = None # Parámetros SMPLX en el mundo "canónico"


def normalize_rgb_tensor(img, imgenet_normalization=True):
    img = img / 255.0
    if imgenet_normalization:
        img = (
            img - torch.tensor(IMG_NORM_MEAN, device=img.device).view(1, 3, 1, 1)
        ) / torch.tensor(IMG_NORM_STD, device=img.device).view(1, 3, 1, 1)
    return img

def get_projection_matrix_numpy(znear, zfar, tanfovx, tanfovy):
    """
    Crea una matriz de proyección con la convención que espera el rasterizador.
    Versión NumPy para usar dentro de la clase.
    """
    P = np.zeros((4, 4))
    
    # Esta es la estructura que nos funcionó antes
    P[0, 0] = 1.0 / tanfovx
    P[1, 1] = 1.0 / tanfovy
    P[2, 2] = (zfar + znear) / (zfar - znear)
    P[2, 3] = - (2 * zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0 # o -1.0 dependiendo de la dirección Z, pero esta es la estructura
    
    return P


class PoseEstimator:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.mhmr_model = load_model(
            os.path.join(model_path, "pose_estimate", "multiHMR_896_L.pt"),
            model_path=model_path,
            device=self.device,
        )
        self.pad_ratio = 0.2
        self.img_size = 896
        self.fov = 60
    
    def to(self, device):
        self.device = device
        self.mhmr_model.to(device)
        return self

    def get_camera_parameters(self):
        K = torch.eye(3)
        # Get focal length.
        focal = self.img_size / (2 * np.tan(np.radians(self.fov) / 2))
        K[0, 0], K[1, 1] = focal, focal

        K[0, -1], K[1, -1] = self.img_size // 2, self.img_size // 2

        # Add batch dimension
        K = K.unsqueeze(0).to(self.device)
        return K

    def img_center_padding(self, img_np):

        ori_h, ori_w = img_np.shape[:2]

        w = round((1 + self.pad_ratio) * ori_w)
        h = round((1 + self.pad_ratio) * ori_h)

        img_pad_np = np.zeros((h, w, 3), dtype=np.uint8)
        offset_h, offset_w = (h - img_np.shape[0]) // 2, (w - img_np.shape[1]) // 2
        img_pad_np[
            offset_h : offset_h + img_np.shape[0] :,
            offset_w : offset_w + img_np.shape[1],
        ] = img_np

        return img_pad_np, offset_w, offset_h

    def _preprocess(self, img_np):

        raw_img_size = max(img_np.shape[:2])

        img_tensor = (
            torch.Tensor(img_np).to(self.device).unsqueeze(0).permute(0, 3, 1, 2)
        )

        _, _, h, w = img_tensor.shape
        scale_factor = min(self.img_size / w, self.img_size / h)
        img_tensor = F.interpolate(
            img_tensor, scale_factor=scale_factor, mode="bilinear"
        )

        _, _, h, w = img_tensor.shape
        pad_left = (self.img_size - w) // 2
        pad_top = (self.img_size - h) // 2
        pad_right = self.img_size - w - pad_left
        pad_bottom = self.img_size - h - pad_top
        img_tensor = F.pad(
            img_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

        resize_img = normalize_rgb_tensor(img_tensor)

        annotation = (
            pad_left,
            pad_top,
            scale_factor,
            self.img_size / scale_factor,
            raw_img_size,
        )

        return resize_img, annotation

    @torch.no_grad()
    def __call__(self, img_path):
        # image_tensor H W C

        img_np = np.asarray(Image.open(img_path).convert("RGB"))

        raw_h, raw_w, _ = img_np.shape

        # pad image for more accurate pose estimation
        img_np, offset_w, offset_h = self.img_center_padding(img_np)
        img_tensor, annotation = self._preprocess(img_np)
        K = self.get_camera_parameters()

        with torch.cuda.amp.autocast(enabled=True):
            target_human = self.mhmr_model(
                img_tensor,
                is_training=False,
                nms_kernel_size=int(3),
                det_thresh=0.3,
                K=K,
                idx=None,
                max_dist=None,
            )
        if not len(target_human) == 1:
            return SMPLXOutput(
                beta=None,
                is_full_body=False,
                msg=(
                    "more than one human detected"
                    if len(target_human) > 1
                    else "no human detected"
                ),
            )

        # check is full body
        pad_left, pad_top, scale_factor, _, _ = annotation
        j2d = target_human[0]["j2d"]
        # tranform to raw image space
        j2d = (
            j2d - torch.tensor([pad_left, pad_top], device=self.device).unsqueeze(0)
        ) / scale_factor
        j2d = j2d - torch.tensor([offset_w, offset_h], device=self.device).unsqueeze(0)

        # scale ratio
        top = j2d[..., 1].min()
        bottom = j2d[..., 1].max()
        full_body_length = bottom - top
        visible_body_length = min(raw_h, bottom) - max(0, top)
        visible_ratio = visible_body_length / full_body_length
        is_full_body = visible_ratio.cpu().item() >= 0.4  # suppose (upper / the lenght of body = 0.4,  4: 6)

        return SMPLXOutput(
            beta=target_human[0]["shape"].cpu().numpy(),
            is_full_body=is_full_body,
            ratio=visible_ratio.cpu().item(),
            msg="success" if is_full_body else "no full-body human detected",
        )

class FullPoseEstimator(PoseEstimator):
    """
    Calcula y devuelve tanto los parámetros de la cámara en el "mundo real"
    como los parámetros SMPLX completos para un mundo "canónico" centrado en la cámara.
    """
    def _build_real_world_camera(self, human_data, img_width, img_height, near_clip=0.1, far_clip=100.0):
        """
        Calcula los parámetros de la cámara en el mundo real, como se hacía originalmente.
        """
        t_subject_in_cam = human_data['transl'].cpu().numpy().flatten()
        rotvec_subject_in_cam = human_data['rotvec'][0].cpu().numpy()
        R_subject_in_cam = ScipyRotation.from_rotvec(rotvec_subject_in_cam).as_matrix()

        R_cam_in_world = R_subject_in_cam.T
        t_cam_in_world = -R_cam_in_world @ t_subject_in_cam

        view_matrix = np.eye(4)
        view_matrix[:3, :3] = R_cam_in_world.T 
        view_matrix[:3, 3] = -R_cam_in_world.T @ t_cam_in_world
        
        fov_rad = np.radians(self.fov)
        tanfovy = np.tan(fov_rad / 2.0)
        tanfovx = tanfovy * (img_width / img_height)
        
        proj_matrix = get_projection_matrix_numpy(near_clip, far_clip, tanfovx, tanfovy)
        proj_matrix = proj_matrix.T 

        return {
            "camera_position": torch.from_numpy(t_cam_in_world).float(),
            "view_matrix": torch.from_numpy(view_matrix).float(),
            "proj_matrix": torch.from_numpy(proj_matrix).float(),
            "tanfovx": float(tanfovx),
            "tanfovy": float(tanfovy),
            "height": int(img_height),
            "width": int(img_width),
        }

    def _extract_smplx_params(self, human_data):
        output = {
            "betas": human_data["shape"],
            "trans": human_data["transl"],
            "root_pose": human_data["rotvec"][0],
            "body_pose": human_data["rotvec"][1:],
            "expression": human_data["expression"],
        }

        return output

    @torch.no_grad()
    def __call__(self, img_path):
        """Sobrescribe el método base para devolver ambos conjuntos de datos."""
        img_np = np.asarray(Image.open(img_path).convert("RGB"))
        raw_h, raw_w, _ = img_np.shape

        img_np_padded, offset_w, offset_h = self.img_center_padding(img_np)
        img_tensor, annotation = self._preprocess(img_np_padded)
        K = self.get_camera_parameters()

        with torch.cuda.amp.autocast(enabled=True):
            target_human = self.mhmr_model(
                img_tensor, is_training=False, nms_kernel_size=int(3), det_thresh=0.3, K=K, idx=None, max_dist=None
            )
        
        if not target_human or not len(target_human) == 1:
            return SMPLXOutput(
                beta=None, is_full_body=False, ratio=None, msg="No human or multiple humans detected"
            )
        
        human_data = target_human[0]
        
        pad_left, pad_top, scale_factor, _, _ = annotation
        j2d = human_data["j2d"]
        j2d = (j2d - torch.tensor([pad_left, pad_top], device=self.device)) / scale_factor
        j2d = j2d - torch.tensor([offset_w, offset_h], device=self.device)
        top = j2d[..., 1].min()
        bottom = j2d[..., 1].max()
        full_body_length = bottom - top
        visible_body_length = min(raw_h, bottom) - max(0, top)
        visible_ratio = visible_body_length / full_body_length if full_body_length > 0 else 0
        is_full_body = visible_ratio.cpu().item() >= 0.4

        # --- CALCULAR AMBOS CONJUNTOS DE DATOS ---
        camera_data_real_world = self._build_real_world_camera(human_data, raw_w, raw_h)
        smplx_params_canonical = self._extract_smplx_params(human_data)

        return SMPLXOutput(
            beta=human_data["shape"][0].cpu().numpy(),
            is_full_body=is_full_body,
            ratio=visible_ratio.cpu().item(),
            camera_data=camera_data_real_world,
            smplx_params=smplx_params_canonical,
            msg="success" if is_full_body else "no full-body human detected",
        )