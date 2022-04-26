import torch
import numpy as np


### Extract camera calibration parameters for a subject
def camera_calibration(subject:int, calibration:str='intrinsic'):
    assert( calibration in ['intrinsic', 'extrinsic', 'project'] ), "calibration needs to be one of 'intrinsic', 'extrinsic' or 'project'"

    filename = 'subject_%d/body/%s.txt' % (subject, calibration)

    with open(filename) as f:
        lines = f.read().splitlines()[3:]

    if calibration == 'intrinsic':
        rows = 3
        cols = 3
    elif calibration == 'extrinsic':
        rows = 4
        cols = 3
    elif calibration == 'project':
        rows = 3
        cols = 4

    camera_id = [int(id[1:]) for id in lines[::rows+1]]

    parameters = lines.copy()
    del parameters[0::rows+1]
    parameters = torch.from_numpy(np.loadtxt(parameters)).cpu()
    parameters = parameters.reshape(len(camera_id), rows, cols)

    return camera_id, parameters


### Extract camera matrices for a subject and camera
def get_camera_parameters(subject:int, camera_index:int):
    assert(camera_index in range(107)), 'camera_index must be an integer in between 0 and 106'
    intr_id, intrinsic = camera_calibration(subject, calibration='intrinsic')
    extr_id, extrinsic = camera_calibration(subject, calibration='extrinsic')
    proj_id, projection = camera_calibration(subject, calibration='project')
    # intr_id = extr_id = proj_id

    cam = intr_id.index(camera_index)

    C = extrinsic[cam][0]
    R = extrinsic[cam][1:]
    T = - torch.matmul(R, C)
    K = intrinsic[cam]

    R = R.T.unsqueeze(0).cpu()
    T = T.unsqueeze(0).cpu()

    f = torch.tensor((K[0,0], K[1,1]), dtype=torch.float32).unsqueeze(0).cpu()
    p = torch.tensor((K[0,2], K[1,2]), dtype=torch.float32).unsqueeze(0).cpu()

    return R, T, f, p
