try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

import os
import torch
import smplx
import numpy as np

from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer import PerspectiveCameras, TexturesUV
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_edge_loss

from utils.smpl_to_smplx import smpl2smplx
from utils.camera_calibration import get_camera_parameters
from utils.renderers import get_renderers
from utils.pointrend_segmentation import get_pointrend_segmentation
from utils.validation import get_split, get_cam_idx_split, validation_score
from utils.vgg_loss import VGGLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Initialize smplx parameters + displacements given an smplx.SMPLXLayer object
def get_init_mesh(smplx_model, subd, requires_grad=True, device:torch.device=device):
    global_orient = torch.nn.Parameter( torch.Tensor([[0, 0, 0]]).to(device), requires_grad=requires_grad )
    transl = torch.nn.Parameter( torch.Tensor([[0, 0, 0]]).to(device), requires_grad=requires_grad )
    body_pose = torch.nn.Parameter( torch.zeros([1, 21, 3]).to(device), requires_grad=requires_grad )
    left_hand_pose = torch.nn.Parameter( torch.zeros([1, 15, 3]).to(device), requires_grad=requires_grad )
    right_hand_pose = torch.nn.Parameter( torch.zeros([1, 15, 3]).to(device), requires_grad=requires_grad )
    jaw_pose = torch.nn.Parameter( torch.zeros([1, 1, 3]).to(device), requires_grad=requires_grad )
    expression = torch.nn.Parameter( torch.zeros([1, 10]).to(device), requires_grad=requires_grad )
    betas = torch.nn.Parameter( torch.zeros([1, 10]).to(device), requires_grad=requires_grad )
    scale = torch.nn.Parameter( torch.Tensor([1.0]).to(device), requires_grad=requires_grad )

    if subd: # number of vertices for 1 subdivision = 41853
        verts_disps = torch.nn.Parameter( torch.zeros([41853, 1]).to(device), requires_grad=requires_grad )

    else:
        num_smplx_verts = smplx_model.get_num_verts() # 10475 vertices for no subdivision
        verts_disps = torch.nn.Parameter( torch.zeros([num_smplx_verts, 1]).to(device), requires_grad=requires_grad )

    texture = torch.nn.Parameter( torch.zeros([1, 1024, 1024, 3]).to(device), requires_grad=requires_grad )

    return global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale, verts_disps, texture


### Given smplx parameters + displacements (optional), construct corresponding mesh
def construct_textured_mesh(smplx_model, texture_uv, global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale, subd, verts_disps=None, device:torch.device=device):
    smplx_faces = torch.Tensor(smplx_model.faces.astype('int')).type(torch.int32).unsqueeze(0).to(device)

    smplx_verts = smplx_model.forward(global_orient=axis_angle_to_matrix(global_orient),
                                      body_pose=axis_angle_to_matrix(body_pose),
                                      left_hand_pose=axis_angle_to_matrix(left_hand_pose),
                                      right_hand_pose=axis_angle_to_matrix(right_hand_pose),
                                      jaw_pose=axis_angle_to_matrix(jaw_pose),
                                      expression=expression, betas=betas)['vertices'].to(device)

    if subd:
        subdivide = SubdivideMeshes()
        smplx_mesh = Meshes(smplx_verts, smplx_faces)
        smplx_mesh = subdivide.forward(smplx_mesh)
        smplx_verts = smplx_mesh.verts_packed().unsqueeze(0)
        smplx_faces = smplx_mesh.faces_packed().unsqueeze(0)

    smplx_mesh = Meshes(smplx_verts * scale + transl, smplx_faces, texture_uv)

    if verts_disps is not None:
        verts_smplx_disp = (smplx_verts * scale) + (smplx_mesh.verts_normals_packed() * verts_disps).unsqueeze(0)
        smplx_mesh = Meshes(verts_smplx_disp + transl, smplx_faces, texture_uv)

    return smplx_mesh


### Get l1 loss for difference between openpose keypoints and smplx joints
def keypoints_loss(smplx_model, subject, pose, global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale, device:torch.device=device):
    # See https://github.com/vchoutas/smplx/blob/master/smplx/vertex_ids.py
    # and https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    openpose_kpts_ix = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    smplx_kpts_ix = [55, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 56, 57, 58 ,59, 60, 61, 62, 63, 64, 65]

    smplx_joints = smplx_model.forward(global_orient=axis_angle_to_matrix(global_orient),
                                      body_pose=axis_angle_to_matrix(body_pose),
                                      left_hand_pose=axis_angle_to_matrix(left_hand_pose),
                                      right_hand_pose=axis_angle_to_matrix(right_hand_pose),
                                      jaw_pose=axis_angle_to_matrix(jaw_pose),
                                      expression=expression, betas=betas)['joints'].to(device)

    kpts_preds = smplx_joints[0][smplx_kpts_ix] * scale + transl # smplx keypoints prediction

    # Extract openpose keypoints for subject and pose
    kpts_filename = 'subject_%d/body/%s/reconstruction/keypoints.txt' % (subject, pose)
    kpts_gt = torch.Tensor( np.loadtxt(kpts_filename)[openpose_kpts_ix] ).to(device) # openpose keypoints ground truth

    l1_loss = torch.nn.L1Loss().to(device)

    return l1_loss(kpts_preds, kpts_gt)


### Get ground truth image and segmentation
def gt_segmentation(camera_loop, subject, pose, rescale_factor):
    camera_idx_list = []
    silh_photo_list = []
    rgb_photo_list = []

    for camera_idx in camera_loop:
        try:
            # Check if camera exists
            get_camera_parameters(subject, camera_idx)
        except:
            print('camera with index %d does not exist' % camera_idx)
            continue

        # Segment person in photo from camera viewpoint
        photo_path = 'subject_%s/body/%s/image/image%s.jpg' % (subject, pose, str(camera_idx).zfill(7))

        # Sometimes camera exists, but still no corresponding photo
        try:
            photo, silh_photo, rgb_photo = get_pointrend_segmentation(photo_path, device=device)
        except:
            print('image image%s.jpg does not exist' % str(camera_idx).zfill(7))
            continue

        silh_photo = silh_photo[0, ::rescale_factor, ::rescale_factor].float().to(device)
        rgb_photo = rgb_photo[0, ::rescale_factor, ::rescale_factor].to(device)

        camera_idx_list.append(camera_idx)
        silh_photo_list.append(silh_photo)
        rgb_photo_list.append(rgb_photo)

    return camera_idx_list, silh_photo_list, rgb_photo_list


### Neural rendering
def neural_renderer(smplx_model, subject:int, pose:str, iterations:int, smplx_uv_path:str, subdivision:bool=False, rescale_factor:int=3, save_path:str=None, validation:bool=False):
    # Segment all photos
    print('segment all photos before neural rendering for subject %d' % subject)

    path_to_imgs = 'subject_%d/body/%s/image' % (subject, pose)
    train_images, val_images = get_split(path_to_imgs)[:2]

    cam_indices = np.random.choice(107, 107, replace=False)
    if validation:
        cam_indices = get_cam_idx_split(train_images, cam_indices)
        print('training images')

    cam_loop = tqdm(cam_indices, total=len(cam_indices))
    camera_idx_list, silh_photo_list, rgb_photo_list = gt_segmentation(cam_loop, subject, pose, rescale_factor)

    if validation:
        val_cam_indices = get_cam_idx_split(val_images, cam_indices)
        print('validation images')
        val_cam_loop = tqdm(val_cam_indices, total=len(val_cam_indices))
        val_camera_idx_list, val_silh_photo_list, val_rgb_photo_list = gt_segmentation(val_cam_loop, subject, pose, rescale_factor)

    # uv coordinates
    obj_mesh = load_obj(smplx_uv_path, load_textures=False)
    faces_uvs = obj_mesh[1].textures_idx.unsqueeze(0).to(device)
    verts_uvs = obj_mesh[2].verts_uvs.unsqueeze(0).to(device)

    img_size = (1080, 1920) # photo resolution
    render_res = ( int(1080/rescale_factor), int(1920/rescale_factor) ) # render resolution

    ## SMPL fitting + Neural rendering
    print('fit new smplx model to provided humbi smpl parameters')
    global_orient, transl, body_pose, betas, scale = smpl2smplx(smplx_model, subject, pose, pose_iterations=200, shape_iterations=100)[:-2]

    left_hand_pose, right_hand_pose, jaw_pose, expression = get_init_mesh(smplx_model, subdivision)[3:7]
    verts_disps, texture = get_init_mesh(smplx_model, subdivision)[-2:]

    # Seperating parameters in different optimizers improves the learning
    geom_lr = 0.0001
    txt_lr = 0.01
    opt_pose_shape = torch.optim.Adam([body_pose, betas], lr=0.01)
    opt_geom = torch.optim.Adam([verts_disps], lr=geom_lr)
    opt_txt = torch.optim.Adam([texture], lr=txt_lr)

    sched_pose_shape = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_pose_shape, patience=5, threshold=0.1, verbose=True)
    sched_geom = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_geom, patience=5, threshold=0.1, verbose=True)
    sched_txt = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_txt, patience=5, threshold=0.1, verbose=True)

    l1_loss = torch.nn.L1Loss().to(device)
    vgg_loss = VGGLoss().to(device)

    add_vgg_loss = False

    print('neural rendering for subject %d' % subject)
    loop = tqdm(total = iterations * len(camera_idx_list))
    for i in range(iterations):
        total_loss = 0.0
        for k, camera_idx in enumerate(camera_idx_list):
            # Extract camera parameters
            R, T, f, p = get_camera_parameters(subject, camera_idx)

            # Construct camera
            cameras = PerspectiveCameras(focal_length=-f, principal_point=p, R=R, T=T, in_ndc=False, image_size=(img_size,), device=device)

            # Construct mesh
            texture_output = torch.clamp(texture, min=0, max=1)
            verts_disps_output = torch.clamp(verts_disps, min=0)
            texture_uv = TexturesUV(maps=texture_output, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
            mesh = construct_textured_mesh(smplx_model, texture_uv, global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale, subdivision, verts_disps_output)

            # Render mesh from camera viewpoint
            silhouette_renderer, phong_renderer = get_renderers(cameras, render_res, device=device)
            phong_render = phong_renderer(mesh)
            silhouette_render = silhouette_renderer(mesh)

            rgb_render = phong_render[0, ..., :3]
            silh_render = silhouette_render[0, ..., 3]

            # Compute loss
            loss = l1_loss(rgb_photo_list[k], rgb_render) + l1_loss(silh_photo_list[k], silh_render) # image loss
            loss += 2.0 * mesh_laplacian_smoothing(mesh, method='cot')
            loss += 10.0 * mesh_edge_loss(mesh)
            loss += keypoints_loss(smplx_model, subject, pose, global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale)

            if add_vgg_loss:
                gt_rgb_vgg = torch.movedim(rgb_photo_list[k], 2, 0)
                rgb_render_vgg = torch.movedim(rgb_render, 2, 0)
                loss += vgg_loss(gt_rgb_vgg, rgb_render_vgg) + vgg_loss(silh_photo_list[k], silh_render) # perceptual loss

            total_loss += float(loss)

            # Backpropagate loss
            opt_pose_shape.zero_grad()
            opt_geom.zero_grad()
            opt_txt.zero_grad()
            loop.set_description('neural rendering loss = %.6f' % loss)
            loss.backward()
            opt_pose_shape.step()
            opt_geom.step()
            opt_txt.step()
            loop.update(1)

        print('neural rendering total training loss for iteration %d : %.6f' % ((i+1), total_loss))
        sched_pose_shape.step(total_loss)
        sched_geom.step(total_loss)
        sched_txt.step(total_loss)

        # Validation
        if validation:
            score = 0
            for k, camera_idx in enumerate(val_camera_idx_list):
                # Validation (TODO : checkout error in utils/validation.py)
                score += validation_score(val_rgb_photo_list[k], val_silh_photo_list[k], rgb_render, silh_render)
            print('neural rendering validation score for iteratrion %d : %.6f' % ((i+1), score))

        if save_path is not None and i%3 == 0 and i > 0:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, 'obj_subj_%d_iter_%d.obj' % (subject, i))
            save_obj(filename, verts=mesh.verts_packed(), faces=mesh.faces_packed(), verts_uvs=verts_uvs[0], faces_uvs=faces_uvs[0], texture_map=texture_output[0])

        # Add VGG loss
        if opt_txt.param_groups[0]['lr'] < 0.1 * txt_lr and not add_vgg_loss:
            add_vgg_loss = True
            print('start adding vgg loss')

        # Early stopping
        if (opt_geom.param_groups[0]['lr'] < 0.01 * geom_lr) and (opt_txt.param_groups[0]['lr'] < 0.01 * txt_lr):
            print('early stopping since low learning rate already reached!\n')
            break

    geometry = global_orient.detach(), transl.detach(), body_pose.detach(), left_hand_pose.detach(), right_hand_pose.detach(), jaw_pose.detach(), expression.detach(), betas.detach(), scale.detach(), verts_disps_output.detach()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, 'obj_subj_%d_final.obj' % subject)
        save_obj(filename, verts=mesh.verts_packed(), faces=mesh.faces_packed(), verts_uvs=verts_uvs[0], faces_uvs=faces_uvs[0], texture_map=texture_output[0])

    return geometry, texture_output.detach()
