try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

import os
import cv2
import torch
import numpy as np
from typing import List

from utils.inpainting import verts_uvs_positions, get_disps_inpaint


### Given the learned geometry for multiple subjects (in npz files under npz_path), compute normalized displacement textures
### i.e., find 2 biggest displacements (negative and positive ones), and map those values linearly to 0 and 255
def normalize_displacements(subjects:List[int], npz_path:str, save_folder:str, smplx_uv_path:str, uv_mask_img:str):
    global_min = 0
    global_max = 0

    print('find normalization range among all displacements')
    for subject in subjects:
        npz_subject_path = os.path.join(npz_path, 'output_subject_%d.npz' % subject)
        if os.path.exists(npz_subject_path):
            learned_geometry = torch.Tensor(np.load(npz_subject_path)['learned_geometry'])
            if learned_geometry.min() < global_min:
                global_min = learned_geometry.min()
            if learned_geometry.max() > global_max:
                global_max = learned_geometry.max()

            nrm_path = os.path.join(npz_path, 'normalization.npz')
            np.savez(nrm_path, global_min=global_min.numpy(), global_max=global_max.numpy())

    print('construct normalized displacement textures')
    save_count = 0
    for subject in tqdm(subjects):
        npz_subject_path = os.path.join(npz_path, 'output_subject_%d.npz' % subject)
        if os.path.exists(npz_subject_path):
            learned_geometry = torch.Tensor(np.load(npz_subject_path)['learned_geometry'])
            nrm_learned_geometry = (learned_geometry - global_min) / (global_max - global_min) # map between 0 and 1
            zero = - global_min / (global_max - global_min)

            disps_x = get_disps_inpaint(subject, nrm_learned_geometry[:,0], smplx_uv_path, uv_mask_img, mask_disps=True, fill_value=zero)[0]
            disps_y = get_disps_inpaint(subject, nrm_learned_geometry[:,1], smplx_uv_path, uv_mask_img, mask_disps=True, fill_value=zero)[0]
            disps_z = get_disps_inpaint(subject, nrm_learned_geometry[:,2], smplx_uv_path, uv_mask_img, mask_disps=True, fill_value=zero)[0]
            displacement_map = torch.cat((disps_x.unsqueeze(2), disps_y.unsqueeze(2), disps_z.unsqueeze(2)), dim=2)

            nrm_displacement_map = (displacement_map.cpu().numpy() * 255).astype(np.uint8)

            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, 'disp_texture_%d.png' % subject)
            save_count += cv2.imwrite(save_path, cv2.cvtColor(nrm_displacement_map, cv2.COLOR_BGR2RGB))

    return save_count


### Given the path to an image and the normalization range (min and max), compute the true displacements (inverse normalization)
def denormalize_disp(path:str, glob_min:int, glob_max:int, smplx_uv_path:str=None):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    denorm_image = (image / 255.0) * (glob_max - glob_min) + glob_min

    # return displacements (1) as image, and (2) as tensor of 3D displacements (per vertex) only if the smplx_uv_path is provided
    if smplx_uv_path is None:
        return denorm_image
    else:
        idx = torch.round(verts_uvs_positions(smplx_uv_path, image.shape[0])).cpu()
        xyz_disps = denorm_image[idx[:,1].long(), idx[:,0].long()]
        return denorm_image, xyz_disps
