import os
from typing import Optional
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import monai
from sklearn.model_selection import train_test_split


class UNetModel:

    def __init__(self, config, train=False):
        super().__init__()
        self.lr_finder = None
        self._train_dataset = None
        self._train_dataloader = None
        self._train_transforms = None
        self._dataset = None
        self._dataloader = None
        self._transforms = None
        self._train = train
        self._optimizer = None
        self._scheduler_cyclic = None
        self._scheduler_exponential = None
        self._loss_function = None
        self._lr_finder = None

        self._model_name = 'unet'
        if 'name' in config:
            self._model_name = config['name'].lower()
            
        self._channels_in = 1
        if 'channels_in' in config:
            self._channels_in = config['channels_in']

        self._channels_out = 2
        if 'channels_out' in config:
            self._channels_out = config['channels_out']

        self._filters = (16, 32, 64, 128)
        if 'filters' in config:
            self._filters = config['filters']

        if len(self._filters) == 2:
            self._strides = (2, )
        if len(self._filters) == 3:
            self._strides = (2, 2)
        if len(self._filters) == 4:
            self._strides = (2, 2, 2)
        if len(self._filters) == 5:
            self._strides = (2, 2, 2, 2)
        if len(self._filters) == 6:
            self._strides = (2, 2, 2, 2, 2)

        self._input_shape = (128, 128, 128)
        if 'input_shape' in config:
            self._input_shape = config['input_shape']
            
        self._roi_shape = None
        if 'roi_shape' in config:
            self._roi_shape = config['roi_shape']

        self._sliding_window_shape = self._roi_shape
        if 'sliding_window_shape' in config:
            self._sliding_window_shape = config['sliding_window_shape']

        self._cuda = True
        if 'cuda' in config:
            self._cuda = config['cuda']

        if self._cuda:
            self._device = 'cuda'
        else:
            self._device = 'cpu'

        self._batch_size = 1
        if 'batch_size' in config:
            self._batch_size = config['batch_size']

        self._sw_batch_size = 1
        if 'sw_batch_size' in config:
            self._sw_batch_size = config['sw_batch_size']

        self._num_res_units = 0
        if 'num_res_units' in config:
            self._num_res_units = config['num_res_units']

        self._cache_size = self._batch_size
        if 'cache_size' in config:
            self._cache_size = config['cache_size']

        self._base_learning_rate = 1e-3
        if 'base_learning_rate' in config:
            self._base_learning_rate = config['base_learning_rate']

        self._train_test_split = .8
        if 'train_test_split' in config:
            self._train_test_split = config['train_test_split']

        self._dropout = .1
        if 'dropout' in config:
            self._dropout = config['dropout']

        if self._model_name == 'unet':
            self._model = monai.networks.nets.Unet(
                spatial_dims=3,
                in_channels=self._channels_in,
                out_channels=self._channels_out,
                channels=self._filters,
                strides=self._strides,
                num_res_units=self._num_res_units,
                # norm=monai.networks.layers.Norm.BATCH,
                norm='instance',
                dropout=self._dropout
            )
        elif self._model_name == 'attentionunet':
            self._model = monai.networks.nets.AttentionUnet(
                spatial_dims=3,
                in_channels=self._channels_in,
                out_channels=self._channels_out,
                channels=self._filters,
                strides=self._strides,
                dropout=.10
            )

        if self._cuda:
            self._model = self._model.cuda()

    def forward(self, x):
        return self._model(x)

    def model(self):
        return self._model

    def loss_function(self):
        return self._loss_function

    def cuda(self):
        return self._cuda

    def device(self):
        return self._device

    def optimizer(self):
        return self._optimizer

    def batch_size(self):
        return self._batch_size

    def sw_batch_size(self):
        return self._sw_batch_size

    def num_res_units(self):
        return self._num_res_units

    def channels_in(self):
        return self._channels_in

    def channels_out(self):
        return self._channels_out

    def input_shape(self):
        return self._input_shape

    def roi_shape(self):
        return self._roi_shape

    def set_train_transforms(self, xforms: monai.transforms.transform):
        self._train_transforms = xforms

    def set_transforms(self, xforms: monai.transforms.transform):
        self._transforms = xforms

    def prepare_training_data(self,
                              images_labels_function: callable,
                              images_folder: str,
                              labels_folder: str,
                              max_files: Optional[int] = None
                              ):

        images_labels = images_labels_function(images_folder, labels_folder)

        if max_files is not None:
            images_labels = images_labels[:max_files]

        if self._train:
            train_images_labels, valid_images_labels = train_test_split(
                images_labels, test_size=0.2, random_state=42
            )
            print(f"Training: {len(train_images_labels)}, Validation: {len(valid_images_labels)}")
        else:
            train_images_labels = None
            valid_images_labels = images_labels

        if self._train:
            print("Building CacheDataset (train) ...")
            self._train_dataset = monai.data.Dataset(data=train_images_labels,
                                                     transform=self._train_transforms)  # , cache_num=self._cache_size)
            print(" ... done Building CacheDataset (train) ..., building dataLoader (train)")
            self._train_dataloader = DataLoader(self._train_dataset, batch_size=self._batch_size,
                                                num_workers=0,
                                                collate_fn=monai.data.pad_list_data_collate)
            print(" ... done Building DataLoader (train).")
        else:
            self._train_dataset = None
            self._train_dataloader = None

        print("Building CacheDataset (valid) ...")
        self._dataset = monai.data.Dataset(data=valid_images_labels,
                                                 transform=self._transforms)  # , cache_num=1)
        print(" ... done Building CacheDataset (valid) ..., building dataLoader (valid)")
        self._dataloader = DataLoader(self._dataset, batch_size=self._batch_size, num_workers=0,
                                            collate_fn=monai.data.pad_list_data_collate)
        print(" ... done Building DataLoader (valid).")

    def initialize_training(self):
        if self._train:

            self._loss_function = monai.losses.DiceCELoss(
                to_onehot_y=False,
                softmax=True,
                include_background=True,
                lambda_dice=0.5,
                lambda_ce=0.5
            )
            self._optimizer = optim.Adam(self._model.parameters(), lr=self._base_learning_rate)
            self._scheduler_cyclic = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer,
                T_0=20,  # first cycle is 10 epochs
                T_mult=2,  # next is 20, then 40, etc.
                eta_min=1e-5
            )

    def scheduler_cyclic(self):
        return self._scheduler_cyclic

    def scheduler_exponential(self):
        return self._scheduler_exponential

    def train_dataloader(self):
        return self._train_dataloader

    def valid_dataloader(self):
        return self._dataloader

    def train_dataset(self):
        return self._train_dataset

    def valid_dataset(self):
        return self._dataset

    def is_using_label_roi(self):
        return True if self._roi_shape is not None else False

    def roi_size(self):
        return self._roi_shape

    def sliding_window_size(self):
        return self._sliding_window_shape

    def set_weights(self, restore_check_point):

        if restore_check_point:
            if os.path.isfile(restore_check_point):
                model_dict = self._model.state_dict()
                pretrained_dict = torch.load(restore_check_point)

                # keep only matching shapes
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if (k in model_dict and model_dict[k].shape == v.shape)
                }

                # update model_dict with pretrained weights
                model_dict.update(pretrained_dict)

                # load the updated weights back into the model
                self._model.load_state_dict(model_dict, strict=False)

        else:
            def weights_init(m):
                class_name = m.__class__.__name__
                if class_name.find('Conv3d') != -1:
                    torch.nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

            self._model.apply(weights_init)


