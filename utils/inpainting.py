import torch
import smplx
import numpy as np

from scipy import interpolate
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Compute vertex uv pixel positions onto a 2D square map
def verts_uvs_positions(smplx_uv_path:str, map_size:int=1024):
    # See https://github.com/facebookresearch/pytorch3d/discussions/588
    smplx_uv_mesh = load_obj(smplx_uv_path, load_textures=False)

    nb_verts = smplx_uv_mesh[0].shape[0]

    # TODO : find pixel vertex uv positions for subdivided smplx mesh
    assert(nb_verts == 10475), "can't find the right vertex uv positions for this .obj file"

    flatten_verts_idx = smplx_uv_mesh[1].verts_idx.flatten().to(device)
    flatten_textures_idx = smplx_uv_mesh[1].textures_idx.flatten().to(device)
    verts_uvs = smplx_uv_mesh[2].verts_uvs.to(device)

    verts_to_uv_index = torch.zeros(nb_verts, dtype=torch.int64).to(device)
    verts_to_uv_index[flatten_verts_idx] = flatten_textures_idx
    verts_to_uvs = verts_uvs[verts_to_uv_index]

    uv_x = ( float(map_size) * verts_to_uvs[:,0] ).unsqueeze(0).to(device)
    uv_y = ( float(map_size) * (1.0 - verts_to_uvs[:,1]) ).unsqueeze(0).to(device)
    verts_uvs_positions = torch.cat((uv_x, uv_y)).moveaxis(0,1).to(device)

    return verts_uvs_positions


### Create displacement map for each vertex and perform interpolation (inpaining) between vertex values
def get_disps_inpaint(subject:int, displacements:torch.Tensor, smplx_uv_path:str, uv_mask_img:str, mask_disps:bool=False, fill_value:int=0):
    uv_mask = read_image(uv_mask_img, mode=ImageReadMode.GRAY_ALPHA)
    # resize = Resize(uv_mask.shape[-1] * 4)
    # uv_mask = resize(uv_mask)
    uv_mask = torch.moveaxis(uv_mask, 0, 2).to(device)
    map_size = uv_mask.shape[:2]

    verts_uvs = verts_uvs_positions(smplx_uv_path, map_size[0]).flip(1)

    mask = (uv_mask[:,:,0] == 0) & (uv_mask[:,:,1] == 0)

    interp = interpolate.LinearNDInterpolator(points=verts_uvs.cpu(), values=displacements.detach().cpu(), fill_value=fill_value)
    inpainted_displacements = interp( list(np.ndindex(map_size)) ).reshape(map_size)

    if mask_disps:
        inpainted_displacements[mask.cpu()] = fill_value

    return torch.Tensor(inpainted_displacements).cpu(), ~mask.cpu()
