try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

import torch
import smplx
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.loss import chamfer_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Extract smpl parameters, and store into smplx compatible parameters tensors
def extract_smpl_param(subject:int, pose:str, device:torch.device=device):
    filename = 'subject_%d/body/%s/reconstruction/smpl_parameter.txt' % (subject, pose)
    smpl_param = np.loadtxt(filename)

    ## See https://github.com/zhixuany/HUMBI#body--cloth for how to extract parameters
    ## and https://github.com/facebookresearch/frankmocap/issues/91 to see smpl pose to smplx

    scale = torch.Tensor([smpl_param[0]]).to(device) # smpl and smplx compatible
    transl = torch.Tensor(smpl_param[1:4]).to(device) # smpl and smplx compatible (no perfect match)
    global_orient = torch.Tensor(smpl_param[4:7]).unsqueeze(0).to(device) # smpl and smplx compatible (no perfect match)
    # body_pose = torch.Tensor(smpl_param[7:76]).reshape(1, 23, 3).to(device) # only smpl compatible
    body_pose_smplx = torch.Tensor(smpl_param[7:70]).reshape(1, 21, 3).to(device) # smplx compatible (no perfect match)
    # betas = torch.Tensor(smpl_param[76:]).unsqueeze(0).to(device) # only smpl compatible

    return scale, transl, global_orient, body_pose_smplx


### Initialize smplx parameters given an smplx.SMPLXLayer object
def get_init_smplx(smplx_model, requires_grad=True, device:torch.device=device):
    global_orient = torch.nn.Parameter( torch.Tensor([[0, 0, 0]]).to(device), requires_grad=requires_grad )
    transl = torch.nn.Parameter( torch.Tensor([[0, 0, 0]]).to(device), requires_grad=requires_grad )
    body_pose = torch.nn.Parameter( torch.zeros([1, 21, 3]).to(device), requires_grad=requires_grad )
    betas = torch.nn.Parameter( torch.zeros([1, 10]).to(device), requires_grad=requires_grad )
    scale = torch.nn.Parameter( torch.Tensor([1.0]).to(device), requires_grad=requires_grad )

    return global_orient, transl, body_pose, betas, scale


### Construct smpl mesh for subject and pose extracted from humbi
def humbi_smpl_mesh(subject:int, pose:str, device:torch.device=device):
    smpl_verts_path = 'subject_%d/body/%s/reconstruction/smpl_vertex.txt' % (subject, pose)
    smpl_faces_path = 'subject_%d/body/smpl_mesh.txt' % subject

    smpl_verts = np.loadtxt(smpl_verts_path)
    smpl_verts = torch.Tensor(smpl_verts).unsqueeze(0).to(device)

    smpl_faces = np.loadtxt(smpl_faces_path)
    smpl_faces = torch.Tensor(smpl_faces).unsqueeze(0).to(device)

    smpl_mesh = Meshes(smpl_verts, smpl_faces)

    return smpl_mesh


### Given smplx parameters (orientation, pose, shape), construct corresponding mesh
def construct_smplx_mesh(smplx_model, global_orient, transl, body_pose, betas, scale, device:torch.device=device):
    smplx_faces = torch.Tensor(smplx_model.faces.astype('int')).type(torch.int32).unsqueeze(0).to(device)

    smplx_verts = smplx_model.forward(global_orient=axis_angle_to_matrix(global_orient),
                                      body_pose=axis_angle_to_matrix(body_pose),
                                      betas=betas)['vertices'].to(device)

    smplx_mesh = Meshes(smplx_verts * scale + transl, smplx_faces)

    return smplx_mesh


### Optimization loop for smpl to smplx parameter learning
def optimization_loop(smpl_mesh, smplx_model, global_orient, transl, body_pose, betas, scale, opt, sched, iters):
    loop = tqdm(range(iters), total = iters)

    for i in loop:
        smplx_mesh = construct_smplx_mesh(smplx_model, global_orient, transl, body_pose, betas, scale)

        sample_smpl = sample_points_from_meshes(smpl_mesh, num_samples=10**3)
        sample_smplx = sample_points_from_meshes(smplx_mesh, num_samples=10**3)

        loss_forward, _ = chamfer_distance(sample_smpl, sample_smplx)
        loss_backward, _ = chamfer_distance(sample_smplx, sample_smpl)
        loss = loss_forward + loss_backward

        opt.zero_grad()
        loop.set_description('smpl2smplx loss = %.6f' % loss)
        loss.backward()
        opt.step()
        sched.step(loss)

    return global_orient, transl, body_pose, betas, scale, loss


### Learn smplx parameters for smpl mesh for subject and pose
def smpl2smplx(smplx_model, subject:int, pose:str, pose_iterations:int=200, shape_iterations:int=100):
    global_orient, transl, body_pose, betas, scale = get_init_smplx(smplx_model)
    scale, transl, global_orient, body_pose = extract_smpl_param(subject, pose)
    scale = torch.nn.Parameter(scale, requires_grad=True)
    transl = torch.nn.Parameter(transl, requires_grad=True)
    global_orient = torch.nn.Parameter(global_orient, requires_grad=True)
    body_pose = torch.nn.Parameter(body_pose, requires_grad=True)

    pose_optimizer = torch.optim.Adam([global_orient, transl, body_pose, scale], lr=0.01)
    pose_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pose_optimizer, patience=20, verbose=True)

    shape_optimizer = torch.optim.Adam([betas], lr=0.1)
    shape_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(shape_optimizer, patience=20, verbose=True)

    smpl_mesh = humbi_smpl_mesh(subject, pose)

    print('fit pose parameters first for subject %d and pose %s' % (subject, pose))
    pose_opt_output = optimization_loop(smpl_mesh, smplx_model, global_orient, transl, body_pose, betas, scale, pose_optimizer, pose_scheduler, pose_iterations)
    global_orient, transl, body_pose, betas, scale, pose_loss = pose_opt_output

    print('then fit shape parameters for subject %d and pose %s' % (subject, pose))
    shape_opt_output = optimization_loop(smpl_mesh, smplx_model, global_orient, transl, body_pose, betas, scale, shape_optimizer, shape_scheduler, shape_iterations)
    global_orient, transl, body_pose, betas, scale, shape_loss = shape_opt_output

    return global_orient, transl, body_pose, betas, scale, pose_loss, shape_loss
