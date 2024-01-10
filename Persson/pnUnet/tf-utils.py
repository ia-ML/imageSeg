import os, time, cv2
from collections import Counter
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
from dataset import CarvanaDataset  # Make sure to adapt CarvanaDataset for TensorFlow as well


def gif2png(inputPath,outputPath):
    #sudo apt install imagemagick
    cmd = 'for file in '+inputPath+'/*.gif; do convert "$file" '+outputPath+'"/${file%.*}.png" done'
    os.system(cmd)

def getClassSize(imgPath,isBin=1):
    img =  cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    imgSize= img.shape
    flattened_img = img.flatten()
    pixel_counts = len(flattened_img)#Counter(flattened_img)
    # print("imgPath       : ",imgPath)
    # print("imgSize       : ",imgSize)    
    # print("pixel_counts : ",pixel_counts)
    if isBin:
       flattened_img[flattened_img>0]=1
    # Count the occurrence of each pixel value
    pixels =Counter(flattened_img)
    # Print pixel values and their counts
    classSizeLst=[]
    for pixel_value, count in pixels.items():
        #print(f"Pixel Value: {pixel_value}, Count: {count}")
        cSize = round(count/pixel_counts,2)
        #print("Pixel Value: ", pixel_value," Count: ", cSize)
        classSizeLst.append([pixel_value,cSize])
    return classSizeLst

def getImagesList(imagePath, rt = [0.70, 0.10, 0.20], doPrint=1):
    fnms = sorted(os.listdir(imagePath))
    img_fnms = sorted([x for x in fnms if not "mask" in x])
    msk_fnms = sorted([x for x in fnms if     "mask" in x])
   
    # msk_fnms = sorted(os.listdir(maskPath))
    # msk_fnms = [x for x in msk_fnms if ".png" in x]   
    numImages = len(img_fnms)
    
    #create training, validation and test sets: 
    #ratios os training, validation, and testing
    
    numTraining   = int(rt[0] * numImages)
    numValidation = int(rt[1] * numImages)
    #numTesting    = int(rt[2] * numImages)
    numTesting    = int(numImages-numTraining-numValidation)
    trnData = [ [x,y] for x,y in zip(img_fnms[:numTraining],msk_fnms[:numTraining])]
    valEnd = numTraining+numImages-numTraining-numTesting
    valData = [ [x,y] for x,y in zip(img_fnms[numTraining:valEnd],msk_fnms[numTraining:valEnd])]
    tstData = [ [x,y] for x,y in zip(img_fnms[numTraining+numValidation:],msk_fnms[numTraining+numValidation:])]
    if doPrint:
        print("img_fnms : ",len(img_fnms))
        print("msk_fnms : ",len(msk_fnms))
        print("Number of training images  : ", len(trnData),numTraining)
        print("Number of validation images: ", len(valData),numValidation)
        print("Number of testing images   : ", len(tstData),numTesting)
        print("Number of images   : ", len(tstData)+len(valData)+len(trnData), numImages)
    imagesLst = [trnData,valData,tstData]   
    
    return imagesLst


def resizeImages(imagePath, maskPath, outputPath, newSize = (480,320) ):   
    ## Resize all images to a smaller size e.g. size/4
    sTime = time.time()
    img_fnms = sorted(os.listdir(imagePath))
    
    msk_fnms = sorted(os.listdir(maskPath))
    ## assuming mixed formats gif and png
    msk_fnms = [x for x in msk_fnms if ".png" in x]

    if not os.path.exists(outputPath):
       os.mkdir(outputPath)
    
    for imgFnm,mskFnm in zip(img_fnms,msk_fnms):
        print(imgFnm,mskFnm)
        imgPath = os.path.join(imagePath,imgFnm)
        mskPath = os.path.join(maskPath,mskFnm)
    
        #resize
        outImg =cv2.resize(cv2.imread(imgPath), newSize)
        outMsk =cv2.resize(cv2.imread(mskPath), newSize)
        #save result
        imgPath = os.path.join(outputPath,imgFnm[:-4]+".png")
        mskPath = os.path.join(outputPath,mskFnm[:-4]+".png")
        cv2.imwrite(imgPath, outImg)
        cv2.imwrite(mskPath, outMsk)
        
    eTime =time.time()
    print("done!!! resize preprocessingTime: ", eTime- sTime," seconds")



def save_checkpoint(model, filepath="my_checkpoint.h5"):
    print("=> Saving checkpoint")
    model.save(filepath)

def load_checkpoint(filepath, custom_objects=None):
    print("=> Loading checkpoint")
    return tf.keras.models.load_model(filepath, custom_objects=custom_objects)

# def get_dataset(image_dir, imgLst, transform):
#     dataset = CarvanaDataset(image_dir, imgLst, transform)
#     return tf.data.Dataset.from_generator(
#         lambda: dataset, 
#         output_types=(tf.float32, tf.float32),
#         output_shapes=([None, None, 3], [None, None])
#     )

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


def check_accuracy(dataset, model):
    num_correct = 0
    num_pixels  = 0
    dice_score  = 0
    num_batches = 0

    for x, y in dataset:
        preds = tf.sigmoid(model(x))
        preds = tf.cast(preds > 0.5, dtype=tf.float32)
        preds = tf.squeeze(preds, axis=-1)
        correct = tf.reduce_sum(tf.cast(preds == y, dtype=tf.int32))
        num_correct += correct
        num_pixels += tf.size(preds).numpy()
        dice = (2 * tf.reduce_sum(preds * y)) / (tf.reduce_sum(preds) + tf.reduce_sum(y) + 1e-8)
        dice_score += dice
        num_batches += 1

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/num_batches}")


import numpy as np
import tensorflow as tf
import os
from PIL import Image

def save_predictions_as_imgs(dataset, model, folder="saved_images/"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for idx, (x, y) in enumerate(dataset):
        preds = tf.sigmoid(model(x))
        preds = tf.cast(preds > 0.5, dtype=tf.float32)
        preds = tf.squeeze(preds, axis=-1)

        # Initialize lists to store concatenated images
        concatenated_preds = []
        concatenated_trues = []

        for i in range(preds.shape[0]):
            pred = tf.keras.preprocessing.image.array_to_img(tf.expand_dims(preds[i], axis=-1))
            true = tf.keras.preprocessing.image.array_to_img(tf.expand_dims(y[i], axis=-1))

            # Convert to numpy array and store
            concatenated_preds.append(np.array(pred))
            concatenated_trues.append(np.array(true))

        # Concatenate images in the batch
        concatenated_preds = np.concatenate(concatenated_preds, axis=1)  # Concatenate along width
        concatenated_trues = np.concatenate(concatenated_trues, axis=1)  # Concatenate along width

        # Convert back to images
        concatenated_preds_img = Image.fromarray(concatenated_preds)
        concatenated_trues_img = Image.fromarray(concatenated_trues)

        # Save the concatenated images
        concatenated_preds_img.save(os.path.join(folder, f"img_{idx}_{i}_pred_k.png"))
        concatenated_trues_img.save(os.path.join(folder, f"img_{idx}_{i}_k.png"))

