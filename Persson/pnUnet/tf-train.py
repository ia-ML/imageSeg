import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import albumentations as A
from tqdm import tqdm
from model import PNUNET  # Ensure this is adapted for TensorFlow
from losses import BinaryDiceLoss  # Ensure this is adapted for TensorFlow

from DNNs.imageSeg.iaUtils.utils import (
    getImagesList,
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# def train_step(model, data, targets, loss_fn, optimizer):
#     with tf.GradientTape() as tape:
#         predictions = model(data, training=True)
#         loss = loss_fn(targets, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss

def train_fn(train_dataset, model, optimizer, loss_fn, DEVICE):
    loop = tqdm(train_dataset)

    for batch_idx, (data, targets) in enumerate(loop):
        with tf.GradientTape() as tape:
            predictions = model(data, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loop.set_postfix(loss=loss.numpy())
 
# Custom function to convert images to TensorFlow tensors
def tf_tensor(image, **kwargs):
    return tf.convert_to_tensor(image)


def main(LEARNING_RATE,DEVICE,BATCH_SIZE,NUM_EPOCHS,NUM_WORKERS,IMAGE_HEIGHT,IMAGE_WIDTH,PIN_MEMORY,LOAD_MODEL,image_dir,output_dir,datasetName,NUM_CLASSES=1):

    print("================================================")
    print("    Segmentation Training Using U-Net TF       ")
    print("================================================")
    
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
    imgLsts = getImagesList(image_dir, rt=[0.70, 0.10, 0.20], doPrint=1)
    trnLst, valLst, tstLst = imgLsts
    train_dataset, val_dataset = get_loaders(image_dir, trnLst, valLst, BATCH_SIZE, train_transform, val_transforms)

    if LOAD_MODEL:
        load_checkpoint(torch.load(finalModelPath), model)

    check_accuracy(val_dataset, model)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_dataset, model, optimizer, loss_fn, DEVICE)

        # save model
        modelPath = os.path.join(output_dir,datasetName+"_"+str(NUM_CLASSES)+"_"+str(epoch)+"_model_pth.keras")
        save_checkpoint(model, modelPath)

        # check accuracy
        check_accuracy(val_dataset, model)

        # print some examples to a folder
        save_predictions_as_imgs( val_dataset, model, folder=output_dir )

    save_checkpoint(model, finalModelPath)
   

if __name__ == "__main__":
    # Hyperparameters etc.
    LEARNING_RATE = 1e-4
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE    = 16
    NUM_EPOCHS    = 3
    NUM_WORKERS   = 2
    IMAGE_HEIGHT  = 160  # 1280 originally
    IMAGE_WIDTH   = 240  # 1918 originally
    PIN_MEMORY    = True
    LOAD_MODEL    = False
    CarvanaPath= "/app/datasets"
    # imagePath = CarvanaPath + "/train_images"
    # maskPath  = CarvanaPath + "/train_masks"
    image_dir    = os.path.join(CarvanaPath,"inputData")
    output_dir   = os.path.join(CarvanaPath,"outputData")
    NUM_CLASSES = 1
    main(LEARNING_RATE,DEVICE,BATCH_SIZE,NUM_EPOCHS,NUM_WORKERS,IMAGE_HEIGHT,IMAGE_WIDTH,PIN_MEMORY,LOAD_MODEL,image_dir,output_dir, NUM_CLASSES)

