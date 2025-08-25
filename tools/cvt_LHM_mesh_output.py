from json import load
from matplotlib.pyplot import sca
import torch
import numpy as np
from plyfile import PlyData, PlyElement
#add the path to sys
import sys,os

# from engine.mmcv.mmcv.parallel import data_container
from LHM.outputs import output
from LHM.outputs.output import GaussianAppOutput
from LHM.models.rendering.utils.sh_utils import RGB2SH, SH2RGB

# front_data=torch.load("/data1/users/wenbo/LHM/exps/Gaussians/Adrian_1/Adrian_1_front_gs_model_list.pth")
# # 'offset_xyz', 'opacity', 'rotation', 'scaling', 'shs', and 'use_rgb'
# front_positions = front_data[0]['offset_xyz'].cpu() #40k,3
# front_scales = front_data[0]['scaling'].cpu()   #40k,3
# front_opacity = front_data[0]['opacity'].cpu()   #40k,1
# front_rotation = front_data[0]['rotation'].cpu() 
# front_shs = front_data[0]['shs'].cpu()         
# front_use_rgb = front_data[0]['use_rgb'] #bool
# #save as new pth file without GaussianAppOutput class

# frong_data_dict = {
#     'offset_xyz': front_positions,
#     'opacity': front_opacity,
#     'rotation': front_rotation,
#     'scaling': front_scales,
#     'shs': front_shs,
#     'use_rgb': front_use_rgb
# }
# torch.save(frong_data_dict, "/data1/users/wenbo/LHM/exps/Gaussians/Adrian_1/front_gs_model_dict.pth")
# print("front_gs_model_dict.pth saved")

# back_positions = back_data[0]['offset_xyz'].cpu().numpy() #40k,3
# back_scales = back_data[0]['scaling'].cpu().numpy()     #40k,3

def inverse_sigmoid(x):

    if isinstance(x, float):
        x = torch.tensor(x).float()

    return torch.log(x / (1 - x))

def construct_list_of_attributes(shs, scaling,rotation):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        features_dc = shs[:, :1]
        features_rest = shs[:, 1:]
        print("features_dc shape",features_dc.shape)
        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

def save_points_to_ply(points, output_file,color=None,radius=None):
    """
    Save points to a PLY file.
    :param points: numpy array of shape (N, 3)
    :param output_file: output PLY file path
    :param color: optional color array of shape (N, 3)
    """
    if color is None:
        vertex_data = np.empty(points.shape[0],
                                 dtype=[('x', 'f4'),
                                          ('y', 'f4'),
                                          ('z', 'f4')])
    else:
        vertex_data = np.empty(len(points),
                       dtype=[('x', 'f4'),
                              ('y', 'f4'),
                              ('z', 'f4'),
                              ('red',   'u1'),
                              ('green', 'u1'),
                              ('blue',  'u1'),])
                            #   ('radius', 'f4')])


    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]
    # if radius is not None:
    #     vertex_data['radius'] = np.mean(radius,axis=1,dtype=np.float32)
    if color is not None:
        vertex_data['red']   = color[:, 0]
        vertex_data['green'] = color[:, 1]
        vertex_data['blue']  = color[:, 2]
    ply_el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([ply_el], text=True).write(output_file)
    print(f"Saved points to {output_file}")

def save_ply(pth_path,output_path):
        
        data=torch.load(pth_path)
        # 'offset_xyz', 'opacity', 'rotation', 'scaling', 'shs', and 'use_rgb'
        xyz = data["offset_xyz"].detach().cpu().numpy()
        print("xyz shape",xyz.shape)
        normals = np.zeros_like(xyz)

        if data["use_rgb"]:
            shs = RGB2SH(data["shs"])
        else:
            shs = data["shs"]

        features_dc = shs[:, :1]
        features_rest = shs[:, 1:]

        f_dc = (
            features_dc.float().detach().flatten(start_dim=1).contiguous().cpu().numpy()
        )
        f_rest = (
            features_rest.float()
            .detach()
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = (
            inverse_sigmoid(torch.clamp(data["opacity"], 1e-3, 1 - 1e-3))
            .detach()
            .cpu()
            .numpy()
        )

        scale = np.log(data["scaling"].detach().cpu().numpy())
        rotation = data["rotation"].detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in construct_list_of_attributes(shs, data["scaling"], rotation)
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(output_path)

def pack_gaussian_asset_and_pos(gs_model_list,position,output_path,input_type):
    position = position.cpu().squeeze(0) #40k,3
    #check if the gs_model_list is instance of GaussianModel
    if input_type=="GaussianModel":
        # self.xyz: Tensor = xyz
        # self.opacity: Tensor = opacity
        # self.rotation: Tensor = rotation
        # self.scaling: Tensor = scaling
        # self.shs: Tensor = shs  # [B, SH_Coeff, 3]
        # self.use_rgb = use_rgb  # shs indicates rgb?
        position_offset = gs_model_list[0].xyz.detach().cpu() #40k,3
        scale = gs_model_list[0].scaling.cpu()   #40k,3
        opacity = gs_model_list[0].opacity.cpu()   #40k,1
        rotation = gs_model_list[0].rotation.cpu()
        shs = gs_model_list[0].shs.cpu()         #40k,1,3
        use_rgb = gs_model_list[0].use_rgb #bool
        save_points_to_ply(position_offset.numpy(),output_path.replace(".pth",".ply"),
                            color=shs.cpu().numpy().squeeze()*255,radius=scale.cpu().numpy())
        
    else:
        position_offset = gs_model_list[0]['offset_xyz'].cpu() #40k,3
        scale = gs_model_list[0]['scaling'].cpu()   #40k,3
        opacity = gs_model_list[0]['opacity'].cpu()   #40k,1
        rotation = gs_model_list[0]['rotation'].cpu()
        shs = gs_model_list[0]['shs'].cpu()         #40k,1,3
        use_rgb = gs_model_list[0]['use_rgb'] #bool
        #save as new pth file without GaussianAppOutput class
        # print("position shape",position.shape)
        # print("position_offset shape",position_offset.shape)
        # print("shs shape",shs.shape)
        save_points_to_ply(position.numpy()+position_offset.cpu().numpy(),output_path.replace(".pth",".ply"),
                            color=shs.cpu().numpy().squeeze()*255,radius=scale.cpu().numpy())
    print("shs shape",shs.shape)
    data_dict={
        "position": position,
        'offset_xyz': position_offset,
        'opacity': opacity,
        'rotation': rotation,
        'scaling': scale,
        'shs': shs,
        'use_rgb': use_rgb
    }
    torch.save(data_dict, output_path)
    print("gs_model.pth saved to ", output_path,"{} points are saved".format(position_offset.shape[0]))

def load_gaussian_asset_from_ply(input_path,output_path=None):

    plydata = PlyData.read(input_path)
    
    if output_path is None:
        output_path = input_path.replace(".ply",".pth")

#mean 3d data
# data=torch.load("/data1/users/wenbo/LHM/exps/Gaussians/Animate_gs_model_before_transform_mean_3d.pth")
# print("load data from Animate_gs_model_before_transform_mean_3d.pth",data.shape)
# save_points_to_ply(data.cpu().numpy(),"/data1/users/wenbo/LHM/exps/Gaussians/Animate_gs_model_before_transform_mean_3d.ply")
# save_ply("/data1/users/adrian/LHM/exps/gaussians/Final_result.pth",
#         "/data1/users/adrian/LHM/exps/gaussians/Final_result.ply")