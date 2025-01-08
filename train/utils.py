from monai.transforms import MapTransform
import os
import shutil
import tempfile
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
# train the model
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric


def validation(model, epoch_iterator_val, config, global_step):
    post_label = AsDiscrete(to_onehot=config["num_classes"])
    post_pred = AsDiscrete(argmax=True, to_onehot=config["num_classes"])
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    model.eval()
    with torch.no_grad():
        dice = []
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), config["input_size"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (128, 128, 128), 1, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            a= dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
            dice.append(a)
        mean_dice_val = dice_metric.aggregate().item()
        print("mean dice score:", torch.nanmean(torch.stack(dice), dim=0))
        dice_metric.reset()
    return mean_dice_val


def train(model, global_step, train_loader,val_loader,config, dice_val_best, global_step_best):

    # loss function and optimizer
    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()

    ## start train the model
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description(f"Training ({global_step} / {config['max_iterations']} Steps) (loss={loss:2.5f})")
        if (global_step % config["eval_num"] == 0 and global_step != 0) or global_step == config["max_iterations"]:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val, config, global_step)
            epoch_loss /= step
            # epoch_loss_values.append(epoch_loss)
            # metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(config["saved_model_dir"], "best_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best