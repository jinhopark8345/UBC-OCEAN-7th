import math
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import pyvips
import torch
import timm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import yangdl as yd
from utils import (
    # CTransPath,
    # DSMIL,
    # Perceiver,
    get_file_names,
    rgb2gray,
    get_biggest_component_box,
)

from model import CTransPath, DSMIL, Perceiver

PATCH_SIZE = 256
THRESH = 0.4

# IMAGES_PATH = '/kaggle/input/UBC-OCEAN/test_images'
IMAGES_PATH = '/kaggle/input/UBC-OCEAN/test_images2'
labels = ['CC', 'EC', 'HGSC', 'LGSC', 'MC', 'Other']

class MyModelModule(yd.ModelModule):
    def __init__(self):
        super().__init__()

        # 1. ctranspath
        self.ctrans = CTransPath(num_classes=0)
        self.ctrans.load_state_dict(torch.load(f'/home/jinho/Projects/UBC-OCEAN-7th/ckpts/ctranspath.pth')['model'], strict=False)
        self.ctrans_dsmils = []
        self.ctrans_perceivers = []
        for fold in (1, 2, 3, 4, 5):
            dsmil = DSMIL(
                num_classes=5,
                size=[768, 128, 128],
                dropout=0.,
            )
            dsmil.load_state_dict(torch.load(f'/home/jinho/Projects/UBC-OCEAN-7th/res/ctrans_dsmil/ckpt/{fold}/best.pt')['model'])
            self.ctrans_dsmils.append(dsmil)
            self.register_models({f'ctrans_dsmil{fold}': dsmil})
        for fold in (1, 2, 3, 4, 5):
            perceiver = Perceiver(
                input_channels=768,
                input_axis=1,
                num_freq_bands=6,
                max_freq=10.,
                depth=1,
                num_latents=1024,
                latent_dim=768,
                cross_heads=1,
                latent_heads=8,
                cross_dim_head=64,
                latent_dim_head=64,
                n_classes=5,
                attn_dropout=0.2,
                ff_dropout=0.2,
                weight_tie_layers=True,
                fourier_encode_data=False,
                self_per_cross_attn=1,
                latent_bounds=2,
                scale=0.125,
            )
            perceiver.load_state_dict(torch.load(f'/home/jinho/Projects/UBC-OCEAN-7th/res/ctrans_perceiver/ckpt/{fold}/best.pt')['model'])
            self.ctrans_perceivers.append(perceiver)
            self.register_models({f'ctrans_perceiver{fold}': perceiver})

        # 2. vits16
        # self.vits16 = torch.load(f'/kaggle/input/ubc-ocean-7th/weights/vits16/1.pt')
        self.vits16 = timm.create_model(model_name="hf-hub:1aurent/vit_small_patch16_224.lunit_dino", pretrained=True)
        self.vits16_dsmils = []
        self.vits16_perceivers = []
        for fold in (1, 2, 3, 4, 5):
            dsmil = DSMIL(
                num_classes=5,
                size=[384, 128, 128],
                dropout=0.,
            )
            dsmil.load_state_dict(torch.load(f'/home/jinho/Projects/UBC-OCEAN-7th/res/vits16_dsmil/ckpt/{fold}/best.pt')['model'])
            self.vits16_dsmils.append(dsmil)
            self.register_models({f'vits16_dsmil{fold}': dsmil})
        for fold in (1, 2, 3, 4, 5):
            perceiver = Perceiver(
                input_channels=384,
                input_axis=1,
                num_freq_bands=6,
                max_freq=10.,
                depth=1,
                num_latents=1024,
                latent_dim=384,
                cross_heads=1,
                latent_heads=8,
                cross_dim_head=64,
                latent_dim_head=64,
                n_classes=5,
                attn_dropout=0.2,
                ff_dropout=0.2,
                weight_tie_layers=True,
                fourier_encode_data=False,
                self_per_cross_attn=1,
                latent_bounds=2,
                scale=0.125,
            )
            perceiver.load_state_dict(torch.load(f'/home/jinho/Projects/UBC-OCEAN-7th/res/vits16_perceiver/ckpt/{fold}/best.pt')['model'])
            self.vits16_perceivers.append(perceiver)
            self.register_models({f'vits16_perceiver{fold}': perceiver})

        self.res = {'image_id': [], 'label': []}

    # x (N, 3, 224, 224)
    def gen_features(self, encoder, x):
        features = []
        for i in range(0, len(x), 64):
            features.append(encoder(x[i: i + 64]))
        features = torch.cat(features, dim=0)

        return features

    def predict_step(self, batch):
        patches, image_id = batch['patches'], batch['image_id']
        patches, image_id = patches[0], image_id[0]

        self.res['image_id'].append(image_id)

        if patches.shape == (0,):
            self.res['label'].append(labels[5])
        else:
            probs = []
            perceiver_probs = []

            """
            total 4 paths,
                - ctrans -> dsmil,
                - ctrans -> perceiver,
                - vits16 -> dsmil,
                - vits16 -> perceiver

            and each path has 5 models,

            so total probs has 20 predictions from each path x #models

            1. perceiver_probs mean (probs from perceiver path of the probs)
                -> if max(mean(perceiver_probs)) < THRESH, select Other label
                -> if max(mean(perceiver_probs)) >= THRESH, select the max(mean(probs)) label
            """

            # ctranspath
            features = self.gen_features(self.ctrans, patches)  # (N, 768)
            for dsmil in self.ctrans_dsmils:
                bag_logits, inst_logits, _, _ = dsmil(features)
                inst_logits, _ = torch.max(inst_logits, dim=0)
                bag_prob = F.softmax(bag_logits, dim=0)
                inst_prob = F.softmax(inst_logits, dim=0)
                probs.append((bag_prob + inst_prob) / 2)
            for perceiver in self.ctrans_perceivers:
                logits, _, _, _, _ = perceiver(features)
                prob = F.sigmoid(logits[0])
                probs.append(prob)
                perceiver_probs.append(prob)

            # vits16
            features = self.gen_features(self.vits16, patches)  # (N, 384)
            for dsmil in self.vits16_dsmils:
                bag_logits, inst_logits, _, _ = dsmil(features)
                inst_logits, _ = torch.max(inst_logits, dim=0)
                bag_prob = F.softmax(bag_logits, dim=0)
                inst_prob = F.softmax(inst_logits, dim=0)
                probs.append((bag_prob + inst_prob) / 2)
            for perceiver in self.vits16_perceivers:
                logits, _, _, _, _ = perceiver(features)
                prob = F.sigmoid(logits[0])
                probs.append(prob)
                perceiver_probs.append(prob)

            probs = torch.stack(probs, dim=0).mean(dim=0)  # (5,)
            pred = probs.argmax(dim=0).item()

            perceiver_probs = torch.stack(perceiver_probs, dim=0).mean(dim=0)  # (5,)
            if max(perceiver_probs) < THRESH:
                pred = 5

            self.res['label'].append(labels[pred])

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.image_ids = get_file_names(IMAGES_PATH, '.png')
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.815, 0.695, 0.808), std=(0.129, 0.147, 0.112)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image, is_tma = self.read_png(f'{IMAGES_PATH}/{image_id}.png')
        patches = self.image2patches(image, patch_size=PATCH_SIZE, step=[256, 64][is_tma], ratio=0.25, transform=self.transform, is_tma=is_tma)

        return {'patches': patches, 'image_id': image_id}

    @staticmethod
    def read_png(image_id: str):
        image = pyvips.Image.new_from_file(image_id, access='sequential').numpy()
        is_tma = image.shape[0] <= 5000 and image.shape[1] <= 5000

        # 1. downsample
        if is_tma:
            resize = A.Resize(image.shape[0] // 4, image.shape[1] // 4)
        else:
            resize = A.Resize(image.shape[0] // 2, image.shape[1] // 2)
        image = resize(image=image)['image']

        # 2. deduplicate for WSI
        if not is_tma:
            resize = A.Resize(image.shape[0] // 16, image.shape[1] // 16)  # downsample for speed
            thumbnail = resize(image=image)['image']
            mask = rgb2gray(thumbnail) > 0
            x0, y0, x1, y1 = get_biggest_component_box(mask)

            # resize box
            scale_h = image.shape[0] / thumbnail.shape[0]
            scale_w = image.shape[1] / thumbnail.shape[1]

            x0 = max(0, math.floor(x0 * scale_w))
            y0 = max(0, math.floor(y0 * scale_h))
            x1 = min(image.shape[1] - 1, math.ceil(x1 * scale_w))
            y1 = min(image.shape[0] - 1, math.ceil(y1 * scale_h))
            image = image[y0: y1 + 1, x0: x1 + 1]

        return image, is_tma

    @staticmethod
    def image2patches(image: np.ndarray, patch_size: int, step: int, ratio: float, transform, is_tma: bool):
        """
        Args:
            image (H, W, 3)

        Returns:
            patches: (N, 256, 256, 3), np.uint8
        """

        patches = []
        for i in range(0, image.shape[0], step):
            for j in range(0, image.shape[1], step):
                patch = image[i: i + patch_size, j: j + patch_size, :]
                if patch.shape != (patch_size, patch_size, 3):
                    patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)))

                if is_tma:
                    patch = transform(image=patch)['image']
                    patches.append(patch)
                else:
                    patch_gray = rgb2gray(patch) # (patch_size, patch_size)
                    patch_binary = (patch_gray <= 220) & (patch_gray > 0)

                    if np.count_nonzero(patch_binary) / patch_binary.size >= ratio:
                        patch = transform(image=patch)['image']
                        patches.append(patch)

        if len(patches) != 0:
            patches = torch.stack(patches, dim=0)
        else:
            patches = torch.zeros(0, dtype=torch.uint8)

        return patches


class MyDataModule(yd.DataModule):
    def __init__(self):
        super().__init__()

    def predict_loader(self):
        dataset = MyDataset()

        yield DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            drop_last=False,
            pin_memory=False
        )

def submit(res):
    pd.DataFrame(res).to_csv('/kaggle/working/submission.csv', index=None)

model_module = MyModelModule()
data_module = MyDataModule()
task_module = yd.TaskModule(model_module, data_module)

task_module.do()
submit(model_module.res)
