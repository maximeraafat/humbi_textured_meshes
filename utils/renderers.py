import torch
import numpy as np
from typing import Union

from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    HardPhongShader,
    SoftPhongShader,
    AmbientLights,
    PointLights,
    BlendParams
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Get silhouette and phong renderers
def get_renderers(cameras, image_size=(360, 640), silh_sigma=1e-7, silh_gamma=1e-1, silh_faces_per_pixel=50, pointlight=False, device:torch.device=device):
    # Settings for opacity and the sharpness of edges
    blend_params = BlendParams(background_color=(0., 0., 0.), sigma=silh_sigma, gamma=silh_gamma)

    # Settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1./1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=silh_faces_per_pixel
    )

    # Create silhouette mesh renderer by composing a rasterizer and a shader
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

    # Also create phong renderer. This is simpler and only needs to render one face per pixel
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1
    )

    blend_params = BlendParams(background_color=(1., 1., 1.))
    lights = AmbientLights(device=device)
    if pointlight:
        lights = PointLights(device=device, location=((2., 2., 2.),))

    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader( # SoftPhongShader
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params
        )
    )

    return silhouette_renderer, phong_renderer
