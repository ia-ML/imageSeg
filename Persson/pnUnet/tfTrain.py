import os, time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import albumentations as A
from tqdm import tqdm
from tfModel import PNUNET  # Ensure this is adapted for TensorFlow
from tfLosses import BinaryDiceLoss  # Ensure this is adapted for TensorFlow

from tfDataset import get_loaders, get_dataset_length

from utils import (
    getImagesList,
    check_accuracy,
    save_predictions_as_imgs,
)

def train_fn(epoch,numEpochs, train_dataset, model, optimizer, loss_fn, DEVICE,n):

    #loop = tqdm(train_dataset)

    trainingLoss = 0.0
    #print("getting length .... ")
    #n = get_dataset_length(train_dataset)
    for batch_idx, (data, targets) in enumerate(train_dataset):
        sTime = time.time()
        #print("getting the gradient tape ....." )
        with tf.GradientTape() as tape:
            #print("getting predictions .... ")
            predictions = model(data, training=True)
            #print("getting loss .... ")
            loss = loss_fn(targets, predictions)
        
        #print("getting gradients .... ")
        gradients = tape.gradient(loss, model.trainable_variables)
        #print("getting optimizer .... ")        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        stepTm = time.time()-sTime
        lossValue = loss.numpy()
        trainingLoss    = trainingLoss + lossValue

        avgTrainingLoss = round(trainingLoss/(batch_idx+1),2)
        line = "epoch: ",epoch+1,"/",numEpochs," idx: ",batch_idx+1,"/",n ," loss: ",round(lossValue,3), \
                        " avg loss: ",avgTrainingLoss, " time: ", round(stepTm,2) 
        #print(line)
        print(f'\r{line}', end='')  # Reprint on the same line
        # update tqdm loop
        #loop.set_postfix(loss=loss.item())
    print()
    trainingLoss = round(trainingLoss/n,2)
    return trainingLoss

# Custom function to convert images to TensorFlow tensors
def tf_tensor(image, **kwargs):
    return tf.convert_to_tensor(image)


def main(LEARNING_RATE,DEVICE,BATCH_SIZE,NUM_EPOCHS,NUM_WORKERS,IMAGE_HEIGHT,IMAGE_WIDTH,
         PIN_MEMORY,LOAD_MODEL,image_dir,output_dir,datasetName,NUM_CLASSES=1,dataRatios=[0.90, 0.10, 0.0]):

    print("================================================")
    print("    Segmentation Training Using U-Net TF       ")
    print("================================================")
    logPath        = os.path.join(output_dir,datasetName+"_"+str(NUM_CLASSES)+"_log.csv")
    finalModelPath = os.path.join(output_dir,datasetName+"_"+str(NUM_CLASSES)+"_model_pth.keras")
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            A.Lambda(image=tf_tensor),  # Convert images to TensorFlow tensors
        ],
    )
   
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            A.Lambda(image=tf_tensor),  # Convert images to TensorFlow tensors
        ],
    )

    # Initialize the model, loss function, and optimizer
    model = PNUNET(in_channels=3, out_channels=NUM_CLASSES)

    # # for multiple classes, we need only to change outchannels to number of classes
    # # and use Crossentropy loss
    # loss_fn = BinaryDiceLoss() if NUM_CLASSES == 1 else tf.keras.losses.SparseCategoricalCrossentropy()    
    loss_fn = BinaryDiceLoss()
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Create TensorFlow datasets for training and validation
    imgLsts = getImagesList(image_dir, rt=dataRatios, doPrint=1)
    trnLst, valLst, tstLst = imgLsts
    train_dataset, val_dataset = get_loaders(image_dir, trnLst, valLst, BATCH_SIZE, train_transform, val_transforms)
    n = len([i for i,x in enumerate(train_dataset)])

    if LOAD_MODEL:
       tf.keras.models.load_model(finalModelPath, custom_objects=model)

    line = "epoch,trainingLoss,validationLoss,Dice,Time"
    with open(logPath, 'w') as file:
            file.write(line + '\n')
    file.close()

    for epoch in range(NUM_EPOCHS):
      
        sTime= time.time()
        trainingLoss = train_fn(epoch,NUM_EPOCHS, train_dataset, model, optimizer, loss_fn, DEVICE,n)
        eTime= round(time.time()-sTime,2)

        valData = []
        for x, y in val_dataset:
            x = x.numpy()
            y = y.numpy()
            preds = tf.sigmoid(model(x))
            preds = tf.cast(preds > 0.5, dtype=tf.float32)
            preds = tf.squeeze(preds, axis=-1)
            p = preds.numpy()
            valData.append([x,y,p])  

        validationLoss = check_accuracy(valData)
        dice = round(1.0-validationLoss,2)

        # print some examples to a folder
        save_predictions_as_imgs( valData,folder=output_dir)

        line = str(epoch+1) +","+ str(trainingLoss) +","+ str(validationLoss) +","+ str(dice) +","+ str(eTime)+","
        print(line)
        with open(logPath, 'a') as file:
                file.write(line + '\n')
        file.close()

        # save model
        modelPath = os.path.join(output_dir,datasetName+"_"+str(NUM_CLASSES)+"_"+str(epoch)+"_model_pth.keras")
        model.save(modelPath)

    model.save(finalModelPath)

if __name__ == "__main__":
    # Hyperparameters etc.
    LEARNING_RATE = 1e-4   
    DEVICE = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    print("DEVICE TensorFlow: ", DEVICE)    
    BATCH_SIZE    = 16
    NUM_EPOCHS    = 3
    NUM_WORKERS   = 2
    IMAGE_HEIGHT  = 160  # 1280 originally
    IMAGE_WIDTH   = 240  # 1918 originally
    PIN_MEMORY    = True
    LOAD_MODEL    = False
    CarvanaPath= "/app/datasets"
    image_dir    = os.path.join(CarvanaPath,"inputData")
    output_dir   = os.path.join(CarvanaPath,"outputData")
    NUM_CLASSES = 1
    dataRatios=[0.10, 0.05, 0.85]

    main(LEARNING_RATE,DEVICE,BATCH_SIZE,NUM_EPOCHS,NUM_WORKERS,IMAGE_HEIGHT,IMAGE_WIDTH,
         PIN_MEMORY,LOAD_MODEL,image_dir,output_dir, NUM_CLASSES,dataRatios)

