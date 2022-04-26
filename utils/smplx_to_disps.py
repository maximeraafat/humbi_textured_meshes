import torch
import smplx
from pytorch3d.structures import Meshes
from pytorch3d.ops import taubin_smoothing, SubdivideMeshes


### Plot interactive scene for list of meshes and/or pointclouds
def smplx2disps(smplx_model, betas, scale, verts_disps, subdivision, smoothing:int=2):
    smplx_faces = torch.Tensor(smplx_model.faces.astype('int')).type(torch.int32).unsqueeze(0).cpu()

    init_verts = smplx_model.forward()['vertices'].cpu() * scale.cpu()
    init_mesh = Meshes(init_verts, smplx_faces)

    displaced_verts = smplx_model.forward(betas=betas)['vertices'].cpu() * scale.cpu()
    displaced_mesh = Meshes(displaced_verts, smplx_faces)

    if subdivision:
        subdivide = SubdivideMeshes()
        init_mesh = subdivide.forward(init_mesh)
        displaced_mesh = subdivide.forward(displaced_mesh)

        displaced_verts = displaced_mesh.verts_packed().unsqueeze(0)
        smplx_faces = displaced_mesh.faces_packed().unsqueeze(0)

    if verts_disps is not None:
        displaced_verts += (displaced_mesh.verts_normals_packed() * verts_disps.cpu()).unsqueeze(0)
        displaced_mesh = Meshes(displaced_verts, smplx_faces)

    if smoothing is not None and smoothing > 0:
        displaced_mesh = taubin_smoothing(displaced_mesh.detach(), num_iter=smoothing)

    displacements = displaced_mesh.verts_packed() - init_mesh.verts_packed()
    displacements /= scale.item()

    # displacements_along_nrm = torch.sum(displacements * init_mesh.verts_normals_packed(), dim=1).cpu()
    # displacements_along_nrm /= scale.item()

    # Plotting issue (plot_structure) if we don't reconstruct the mesh here? Why??
    displaced_mesh = Meshes(displaced_mesh.verts_padded(), displaced_mesh.faces_padded())

    return displacements, init_mesh, displaced_mesh
