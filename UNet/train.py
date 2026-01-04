import argparse
import shutil
import time
import json

import monai.transforms

from utils import *
from drawing_utils import *
from unet_model import *

from Transforms.LoadIXIFreeSurferPair.find_ixi_pairs_v1 import find_ixi_pairs_v1
from UNet.unet_transforms import IXI_Tumors_UNet_Transforms

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pt'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pt')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file-name', type=str, default='config.json')
    parser.add_argument('--epochs', type=int, default=0)
    args = parser.parse_args()

    config_json = None
    try:
        with open(args.config_file_name) as config_json_file:
            config_json = json.load(config_json_file)
    except Exception as e:
        print("Error loading config file: ", e)
        exit(-1)

    is_training = args.epochs > 0

    nn = UNetModel(config_json['model'], train=is_training)
    xforms = IXI_Tumors_UNet_Transforms(
        nn.input_shape(),
        nn.roi_size(),
        nn.channels_in(),
        nn.channels_out()
    )
    nn.set_train_transforms(xforms.train_transforms())
    nn.set_transforms(xforms.valid_transforms())

    # restore weights
    restore_check_point = None
    if 'checkpoints' in config_json and 'restore' in config_json['checkpoints']:
        restore_check_point = config_json['checkpoints']['restore']
    save_check_point = 'model_pt_ct.pth'
    if 'checkpoints' in config_json and 'save' in config_json['checkpoints']:
        save_check_point = config_json['checkpoints']['save']

    nn.set_weights(restore_check_point)

    # load data
    images_folder = config_json['dataset']['images_folder']
    labels_folder = config_json['dataset']['labels_folder']
    max_files = None if is_training else nn.batch_size()
    if 'max_files' in config_json['dataset']:
        max_files = config_json['dataset']['max_files']

    nn.prepare_training_data(
        find_ixi_pairs_v1,
        images_folder=images_folder,
        labels_folder=labels_folder,
        max_files=max_files
    )
    nn.initialize_training()

    post_pred = monai.transforms.Compose([
        monai.transforms.Activations(softmax=True),
        monai.transforms.AsDiscrete(argmax=True, to_onehot=nn.channels_out()),
    ])
    post_label = lambda x: x

    dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean")
    best_metric = -1
    best_metric_epoch = -1

    for epoch in range(args.epochs):
        if epoch == 0:
            print(f"Starting ...")

        train_loss = 0.0
        for b, batch in enumerate(nn.train_dataloader()):
            if b == 0 and epoch == 0:
                print(f"Successfully loaded batch 0.")

            data = batch['image']
            target = batch['labels']
            if nn.cuda():
                data, target = data.cuda(), target.cuda()

            if nn.is_using_label_roi():
                keep = np.arange(data.shape[0]).astype(dtype=np.int32)
                for i in range(data.shape[0]):
                    if torch.count_nonzero(target[i]) == 0:
                        keep = keep[keep != i]
                keep = keep.tolist()
                if len(keep) == 0:
                    continue

                target = target[keep]
                data = data[keep]

            nn.model().train(True)
            nn.optimizer().zero_grad()
            output = nn.forward(data)

            if b == 0 and epoch == 0:
                print(f"Successfully forward'ed batch 0.")

            loss = nn.loss_function()(output, target)
            loss.backward()
            train_loss += loss.data.item()
            nn.optimizer().step()

            if b == 0 and epoch == 0:
                print(f"Successfully backward'ed batch 0.")

            print("Training Epoch, ", epoch+1, " batch:", b+1, " of ", len(nn.train_dataloader()), ", loss", loss.data.item(), train_loss)
            slice = random.randint(0, data.shape[0]-1)
            draw_data_target_output(data.detach().cpu().numpy(), target.detach().cpu().numpy(), output.detach().cpu().numpy(), slice)
            cv.waitKey(1)

        # only validate and save every 10 epochs
        if (epoch + 1) % 10:
            continue

        torch.save(nn.model().state_dict(), "unet_checkpoint_" + str(epoch) + ".pth")

        nn.scheduler_cyclic().step()
        # nn.scheduler_exponential().step()

        # validation set
        with torch.no_grad():
            # metric_values = []

            for b, batch in enumerate(nn.valid_dataloader()):

                nn.model().eval()

                data = batch['image']
                target = batch['labels']
                if nn.cuda():
                    data, target = data.cuda(), target.cuda()

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
                val_outputs = [post_pred(o) for o in monai.data.decollate_batch(val_outputs)]
                val_labels = [post_label(l) for l in monai.data.decollate_batch(target)]

                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

                print("Validating, batch ", b+1, " of ", len(nn.valid_dataloader()))

            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            # metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(nn.model().state_dict(), save_check_point)
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

    # when we're done (or if testing with num_epoch == 0), draw in napari
    import napari
    viewer = napari.Viewer()

    with torch.no_grad():
        nn.model().eval()
        for b, batch in enumerate(nn.valid_dataloader()):

            data = batch['image']
            target = batch['labels']
            if nn.cuda():
                data, target = data.cuda(), target.cuda()

            test_outputs = monai.inferers.sliding_window_inference(
                data,
                nn.sliding_window_size(),
                nn.sw_batch_size(),
                nn.model()
            )
            test_outputs = [post_pred(o) for o in monai.data.decollate_batch(test_outputs)]
            test_labels = [post_label(l) for l in monai.data.decollate_batch(target)]

            # Image.
            viewer.add_image(
                data[0].cpu().numpy()[0]
            )

            colors = [
                (1, 0, 0, 1),
                (0, 1, 0, 1),
                (0, 0, 1, 1),
                (1, 1, 0, 1),
                (1, 0, 1, 1),
                (0, 1, 1, 1),
                (1, 0.5, 0, 1),
                (0.5, 0, 1, 1),
                (0, 0, 0, 1),
                (1, 1, 1, 1),
            ]

            test_output = np.argmax(test_outputs[0].cpu().numpy(), axis=0)
            viewer.add_labels(
               test_output,
               name=f"predicted",
               # color={1: colors[l]}
            )

            napari.run()




if __name__ == '__main__':
    main()
