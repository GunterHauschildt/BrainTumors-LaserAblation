import monai
from monai.inferers import sliding_window_inference
from UNet.unet_model import UNetModel
from UNet.unet_transforms import IXI_Tumors_UNet_Transforms

import torch
import json


class SegmentTumorAndStructureTransform(monai.transforms.Transform):
    def __init__(self, config_file_name, device='cuda'):
        config_json = None
        try:
            with open(config_file_name) as config_json_file:
                config_json = json.load(config_json_file)
        except Exception as e:
            print("Error loading config file: ", e)
            exit(-1)

        self._nn = UNetModel(config_json['model'], train=False)
        xforms = IXI_Tumors_UNet_Transforms(
            self._nn.input_shape(),
            self._nn.roi_size(),
            self._nn.channels_in(),
            self._nn.channels_out()
        )
        self._nn.set_transforms(xforms.test_transforms())

        # restore weights
        restore_check_point = None
        if 'checkpoints' in config_json and 'restore' in config_json['checkpoints']:
            restore_check_point = config_json['checkpoints']['restore']
            self._nn.set_weights(restore_check_point)

        self._post_pred = monai.transforms.Compose([
            monai.transforms.Activations(softmax=True),
            monai.transforms.AsDiscrete(argmax=True, to_onehot=self._nn.channels_out()),
        ])

    def __call__(self, data):

        img = data["image"]   # (1,D,H,W) numpy or torch
        img = torch.as_tensor(img, dtype=torch.float32, device=self._nn.device())[None][None]

        with torch.no_grad():
            output = sliding_window_inference(
                img,
                roi_size=(128, 128, 128),  #   self._nn.roi_shape(),
                sw_batch_size=self._nn.sw_batch_size(),
                predictor=self._nn.model(),
            )
            output = [self._post_pred(o) for o in monai.data.decollate_batch(output)]

        data["labels"] = output[0]
        return data
