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

class ConvertLabelsToZero(MapTransform):
    """
    This custom transformation will convert specified labels to zero.
    """
    def __init__(self, keys, labels_to_zero):
        super().__init__(keys)
        self.labels_to_zero = labels_to_zero

    def __call__(self, data):
        for key in self.keys:
            for label in self.labels_to_zero:
                data[key][data[key] == label] = 0
        return data

class ConvertPixelValues(MapTransform):
    """
    A custom MONAI transform that applies pixel value conversions according to specified mappings.
    """
    def __init__(self, keys, conversions):
        super().__init__(keys)
        self.conversions = conversions

    def __call__(self, data):
        for key in self.keys:
            tensor = data[key]
            for original, new in self.conversions.items():
                tensor[tensor == original] = new
            data[key] = tensor
        return data


def data_transformers(num_samples, device):
  train_transforms = Compose(
      [
          LoadImaged(keys=["image", "label"], ensure_channel_first=True),
          ScaleIntensityRanged(
              keys=["image"],
              a_min=-175,
              a_max=250,
              b_min=0.0,
              b_max=1.0,
              clip=True,
          ),
          CropForegroundd(keys=["image", "label"], source_key="image"),
          Orientationd(keys=["image", "label"], axcodes="RAS"),
          Spacingd(
              keys=["image", "label"],
              pixdim=(1.5, 1.5, 2.0),
              mode=("bilinear", "nearest"),
          ),
          EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
          ConvertLabelsToZero(keys=["label"], labels_to_zero=[5,9,10,12,13]),  # remove small organs
          ConvertPixelValues(keys=["label"], conversions={6: 5, 7: 6, 8: 7, 11: 8}),  # adjust class numbers
          RandCropByPosNegLabeld(
              keys=["image", "label"],
              label_key="label",
              spatial_size=(128, 128, 128),
              pos=1,
              neg=1,
              num_samples=num_samples,
              image_key="image",
              image_threshold=0,
          ),
          RandFlipd(
              keys=["image", "label"],
              spatial_axis=[0],
              prob=0.10,
          ),
          RandFlipd(
              keys=["image", "label"],
              spatial_axis=[1],
              prob=0.10,
          ),
          RandFlipd(
              keys=["image", "label"],
              spatial_axis=[2],
              prob=0.10,
          ),
          RandRotate90d(
              keys=["image", "label"],
              prob=0.10,
              max_k=3,
          ),
          RandShiftIntensityd(
              keys=["image"],
              offsets=0.10,
              prob=0.50,
          ),
      ]
  )
  val_transforms = Compose(
      [
          LoadImaged(keys=["image", "label"], ensure_channel_first=True),
          ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
          CropForegroundd(keys=["image", "label"], source_key="image"),
          Orientationd(keys=["image", "label"], axcodes="RAS"),
          Spacingd(
              keys=["image", "label"],
              pixdim=(1.5, 1.5, 2.0),
              mode=("bilinear", "nearest"),
          ),
          EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
          ConvertLabelsToZero(keys=["label"], labels_to_zero=[5,9,10,12,13]),  # remove small organs
          ConvertPixelValues(keys=["label"], conversions={6: 5, 7: 6, 8: 7, 11: 8}),  # adjust class numbers
      ]
  )
  return train_transforms, val_transforms


def data_loaders(data_dir, num_samples, device):
  split_json = "dataset.json"
  datasets = data_dir + split_json
  train_transforms, val_transforms = data_transformers(num_samples, device)
  datalist = load_decathlon_datalist(datasets, True, "training")
  val_files = load_decathlon_datalist(datasets, True, "validation")
  train_ds = CacheDataset(
      data=datalist,
      transform=train_transforms,
      cache_num=24,
      cache_rate=1.0,
      num_workers=8,
  )
  train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
  val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
  val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
  set_track_meta(False)
  return val_loader, train_loader



