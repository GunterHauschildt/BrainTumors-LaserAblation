import argparse
import shutil
import time
import json

import monai.transforms

from utils import *
from drawing_utils import *
from unet_model import *

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

#
# def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pt'):
#     prefix_save = os.path.join(path, prefix)
#     name = prefix_save + '_' + filename
#     torch.save(state, name)
#     if is_best:
#         shutil.copyfile(name, prefix_save + '_model_best.pt')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file-name', type=str, default='config.json')
    args = parser.parse_args()

    config_json = None
    try:
        with open(args.config_file_name) as config_json_file:
            config_json = json.load(config_json_file)
    except Exception as e:
        print("Error loading config file: ", e)
        exit(-1)

    # define model
    nn = UNetModel(config_json['model'], train=False)

    # restore weights
    restore_check_point = None
    if 'checkpoints' in config_json and 'restore' in config_json['checkpoints']:
        test_check_point = config_json['checkpoints']['test']
    else:
        print(f"Couldn't find checkpoints.test in config file.")
        exit(-1)

    nn.set_weights(test_check_point)

    # load data
    folder = config_json['dataset']['folder']
    max_files = None
    if 'max_files' in config_json['dataset']:
        max_files = config_json['dataset']['max_files']

    nn.prepare_training_data(folder, max_files=max_files)

    # best_valid_loss = np.inf
    # num_epochs = 640
    # nn.initialize_training()
    # post_pred = monai.transforms.Compose([monai.transforms.AsDiscrete(argmax=True, to_onehot=2)])
    # post_label = monai.transforms.Compose([monai.transforms.AsDiscrete(to_onehot=2)])
    # dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean")
    # best_metric = -1
    # best_metric_epoch = -1

    nn.model().eval()
    with torch.no_grad():
        # metric_values = []

        for b, batch in enumerate(nn.valid_dataloader()):

            data = batch['image']
            target = batch['label']
            if nn.cuda():
                data, target = data.cuda(), target.cuda()

            # to do: this isn't clear
            if nn.is_using_label_roi():
                roi_size = nn.sliding_window_size()
                sw_batch_size = nn.sw_batch_size()
            else:
                roi_size = (-1, -1, -1)
                sw_batch_size = 1

            val_outputs = monai.inferers.sliding_window_inference(
                data,
                roi_size,
                sw_batch_size,
                nn.model()
            )
            # val_outputs = [post_pred(i) for i in monai.data.decollate_batch(val_outputs)]
            # val_labels = [post_label(i) for i in monai.data.decollate_batch(target)]

            set_windows()
            slice = random.randint(0, data.shape[0] - 1)
            draw_data_target_output(
                data.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                val_outputs.detach().cpu().numpy(), slice)
            cv.waitKey(1)
        destroy_windows()

if __name__ == '__main__':
    main()
