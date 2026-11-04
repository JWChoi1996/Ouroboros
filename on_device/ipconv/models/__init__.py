from importlib import import_module

from .fasterrcnn_resnet50_fpn import FasterRCNN_ResNet50_FPN_Contexted
from .ViTDet.maskedrcnn_vit_b_fpn import MaskedRCNN_ViT_B_FPN_Contexted
from .ViTDet.maskedrcnn_vit_l_fpn import MaskedRCNN_ViT_L_FPN_Contexted
from .ViTDet.maskedrcnn_vit_h_fpn import MaskedRCNN_ViT_H_FPN_Contexted
from .ViTDet.cascade_mask_rcnn_swin_b import CascadeMaskRCNN_Swin_B_Contexted
from .ViTDet.cascade_mask_rcnn_swin_l import CascadeMaskRCNN_Swin_L_Contexted
from .ViTDet.vit_b_imagenet import ViTDeT_b_Imagenet_Contexted
from .ViTDet.cascade_mask_rcnn_mvit_b import CascadeMaskRCNN_MViT_B_Contexted


_OPTIONAL_MODELS = {
    "DINO_4Scale_Swin_Contexted": (
        ".DINO.dino_4scale_swin",
        "DINO_4Scale_Swin_Contexted",
    ),
    "DINO_5Scale_Swin_Contexted": (
        ".DINO.dino_5scale_swin",
        "DINO_5Scale_Swin_Contexted",
    ),
    "LWDETR_xLarge_Contexted": (
        ".LW_DETR.lwdetr_xlarge",
        "LWDETR_xLarge_Contexted",
    ),
}


def __getattr__(name):
    if name not in _OPTIONAL_MODELS:
        raise AttributeError(name)
    module_name, class_name = _OPTIONAL_MODELS[name]
    value = getattr(import_module(module_name, __name__), class_name)
    globals()[name] = value
    return value