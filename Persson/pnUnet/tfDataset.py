import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from PIL import Image

class CarvanaDataset:
    def __init__(self, image_dir, imgFnmsLst, transform=None):
        self.image_dir = image_dir
        self.imgFnmsLst = imgFnmsLst
        self.transform = transform
        # if offline augmentation:
        # augment the data here 
        
    def __len__(self):
        return len(self.imgFnmsLst)

    def __getitem__(self, index):
        img_path  = os.path.join(self.image_dir, self.imgFnmsLst[index][0])
        mask_path = os.path.join(self.image_dir, self.imgFnmsLst[index][1])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask < 128.0] = 0.0
        mask[mask >= 128.0] = 1.0

        # Correctly pass image and mask to the transform
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask
        
def load_dataset(dataset):
    def generator():
        for i in range(len(dataset)):
            yield dataset[i]

    return tf.data.Dataset.from_generator(
        generator, 
        output_types=(tf.float32, tf.float32), 
        output_shapes=([None, None, 3], [None, None])
    )

def get_loaders(image_dir, trnLst, valLst, batch_size, train_transform, val_transform):
    train_ds = tf.data.Dataset.from_generator(
        lambda: CarvanaDataset(image_dir, trnLst, train_transform),
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, None, 3], [None, None])
    )
    val_ds = tf.data.Dataset.from_generator(
        lambda: CarvanaDataset(image_dir, valLst, val_transform),
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, None, 3], [None, None])
    )

    train_ds = train_ds.batch(batch_size).shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# we can not get the length directly 
def get_dataset_length(data):
   idx = 0
   for x  in data:
       idx = idx +1 
   return idx    


# Example usage
# carvana_dataset = CarvanaDataset(image_dir, imgFnmsLst, transform)
# tf_dataset = load_dataset(carvana_dataset)
# for image, mask in tf_dataset:
#     # process image and mask
