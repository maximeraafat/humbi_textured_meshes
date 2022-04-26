import os
import torch
import numpy as np
from utils.vgg_loss import VGGLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Split the images into training, validation and testing
def get_split(path_to_imgs):
    images = np.asarray(sorted([os.path.join(path_to_imgs, img) for img in os.listdir(path_to_imgs)]))
    n_images = len(images)

    np.random.seed(13)
    images = images[np.random.choice(n_images, n_images, replace=False)]
    np.random.seed(None)

    n_train_images = int(0.8 * n_images)
    n_val_images = int(0.1 * n_images)

    train_images = images[0 : n_train_images]
    val_images = images[n_train_images : n_train_images + n_val_images]
    test_images = images[n_train_images + n_val_images:]
    n_test_images = len(test_images)

    return train_images, val_images, test_images


### Find correspondance between all camera indices and the indices of the images in the split
def get_cam_idx_split(data_split, camera_indices):
    # e.g. for 'subject_1/body/00000001/image/image0000010.jpg', extract '10' as integer
    data_indices = np.array([ int(path.split('image')[-1].split('.jpg')[0]) for path in data_split ])
    data_indices = np.intersect1d(data_indices, camera_indices)
    np.random.shuffle(data_indices)

    return data_indices


### Validation (or test) score for 4 images : the renders and ground truth (rgb + segmentation)
def validation_score(rgb_photo, silh_photo, rgb_render, silh_render):
    l1_loss = torch.nn.L1Loss().to(device)
    vgg_loss = VGGLoss().to(device)

    gt_rgb_vgg = torch.movedim(rgb_photo, 2, 0)
    rgb_render_vgg = torch.movedim(rgb_render, 2, 0)

    score = l1_loss(rgb_photo, rgb_render) + l1_loss(silh_photo, silh_render) # image loss
    # TODO : CUDA RuntimeError after few iterations?
    # score += vgg_loss(gt_rgb_vgg, rgb_render_vgg) + vgg_loss(silh_photo, silh_render) # perceptual loss

    return score
