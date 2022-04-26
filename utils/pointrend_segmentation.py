### See PointRend in detectron2 : https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend

import torch
from torchvision.io import read_image

# setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# create detectron2 config and load PointRend model
cfg = get_cfg()
point_rend.add_pointrend_config(cfg)

# load config from file
cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this model

# use PointRend model zoo
cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
cfg.MODEL.DEVICE = device.type

predictor = DefaultPredictor(cfg)


### PointRend segmentation for an image located in img_path
def get_pointrend_segmentation(img_path, predictor=predictor, class_id=0, device:torch.device=device):
    img = read_image(img_path).permute(1, 2, 0).to(device)
    outputs = predictor(img[:,:,[2,1,0]].cpu().numpy())

    person_ix = (outputs['instances'].pred_classes==class_id).nonzero()[0]
    img_mask = outputs['instances'][person_ix].pred_masks.detach().to(device)

    img_segmented = (img / 255).unsqueeze(0).to(device)
    img_segmented[img_mask==0] = 1

    return img, img_mask, img_segmented
