# original folder structure
# - train_images
# - train_masks
# - val_images
# - val_masks
import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CarvanaDataset(Dataset):
    # def __init__(self, image_dir, mask_dir, transform=None):
    #     self.image_dir = image_dir
    #     self.mask_dir  = mask_dir
    #     self.transform = transform
    #     self.images = os.listdir(image_dir)
    def __init__(self, image_dir, imgFnmsLst, transform=None,
                      datasetStructure=0, segExtension="mask", onlineAugmentation=1):
        self.image_dir = image_dir
        # [[train0,mask0],[train1,mask1],...,etc]
        self.imgFnmsLst  = imgFnmsLst
        self.transform = transform
        #self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.imgFnmsLst)

    def __getitem__(self, index):
        img_path  = os.path.join(self.image_dir, self.imgFnmsLst[index][0])
        mask_path = os.path.join(self.image_dir, self.imgFnmsLst[index][1])
        image   = np.array(Image.open(img_path).convert("RGB"))

        mask      = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask <  128.0] = 0.0
        mask[mask >= 128.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask  = augmentations["mask"]
        return image, mask

def get_loaders(
    image_dir,
    trnLst,
    valLst,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    datasetStructure=0,
    segExtension="mask",
    onlineAugmentation=1
):
    
    train_ds = CarvanaDataset(
        image_dir  = image_dir,
        imgFnmsLst = trnLst, 
        transform = train_transform,
        datasetStructure=0,
        segExtension="mask",
        onlineAugmentation=1
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir = image_dir,
        imgFnmsLst = valLst, 
        transform = val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle    = False,
    )

    return train_loader, val_loader

