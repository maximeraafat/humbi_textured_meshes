import torch
from typing import List, Union
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.structures import Meshes, Pointclouds


### Plot interactive scene for list of meshes and/or pointclouds
def plot_structure(structures:List[ Union[Meshes, Pointclouds] ]):
    assert(bool(structures) != None), 'nothing to be plotted'

    if not isinstance(structures, list):
        structures_clone = [structures.clone().cpu()]
    else:
        structures_clone = []
        for structure in structures:
            structures_clone.append(structure.clone().cpu())

    end = len(structures_clone)
    offsets = torch.arange(0, end, step=1)

    dict_string = []
    for i, structure in enumerate(structures_clone):
        offset = torch.Tensor( [offsets[i], 0, 0] ).cpu()
        if isinstance(structure, Meshes):
            structure.verts_list()[0] = structure.verts_list()[0] + offset
            dict_string.append('mesh %d' % (i+1))
        elif isinstance(structure, Pointclouds):
            structure.points_list()[0] = structure.points_list()[0] + offset
            dict_string.append('pointcloud %d' % (i+1))

    zip_iterator = zip(dict_string, structures_clone)
    plot_structures = {'PLOT': dict(zip_iterator)}

    return plot_scene(plot_structures)
