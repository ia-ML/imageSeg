import os, time
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from ptModel import PNUNET
from ptLosses import BinaryDiceLoss

from ptDataset import get_loaders

from utils import (
    getImagesList,
    check_accuracy,
    save_predictions_as_imgs,
    plotLoss
)


def train_fn(epoch,numEpochs, loader, model, optimizer, loss_fn, scaler, DEVICE):
    #loop = tqdm(loader)
    trainingLoss = 0.0
    #for batch_idx, (data, targets) in enumerate(loop):
    for batch_idx, (data, targets) in enumerate(loader):    
        sTime = time.time()
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        stepTm = time.time()-sTime
        trainingLoss    = trainingLoss + loss.item()
        avgTrainingLoss = round(trainingLoss/(batch_idx+1),2)
        line = "epoch: ",epoch+1,"/",numEpochs," idx: ",batch_idx+1,"/",len(loader) ," loss: ",round(loss.item(),3), \
                        " avg loss: ",avgTrainingLoss, " time: ", round(stepTm,2) 
        #print(line)
        print(f'\r{line}', end='')  # Reprint on the same line
        # update tqdm loop
        #loop.set_postfix(loss=loss.item())
    print()
    trainingLoss = round(trainingLoss/len(loader),2)
    return trainingLoss

def getImageTransforms(IMAGE_HEIGHT,IMAGE_WIDTH,onlineAugmentation):
   # Create the main transformation pipeline        
    normalizeTransform =  A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        )
    augTransform = [ A.Rotate(limit=35, p=1.0),
                     A.HorizontalFlip(p=0.5),
                     A.VerticalFlip(p=0.1)]
    tensorTransform = ToTensorV2()  # Convert images to tensors

    train_transform = []
    val_transform   = []
    # Check if IMAGE_HEIGHT and IMAGE_WIDTH are not None
    if IMAGE_HEIGHT is not None and IMAGE_WIDTH is not None:
        train_transform.append(A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH))
        val_transform.append(A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH))
    if   onlineAugmentation:
         train_transform.extend(augTransform)

    train_transform.extend([normalizeTransform,tensorTransform])
    val_transform.extend([normalizeTransform,tensorTransform])


    train_transforms = A.Compose(train_transform)
    val_transforms   = A.Compose(val_transform)
    return train_transforms, val_transforms

def main(LEARNING_RATE,DEVICE,BATCH_SIZE,NUM_EPOCHS,NUM_WORKERS,IMAGE_HEIGHT,IMAGE_WIDTH,
         PIN_MEMORY,LOAD_MODEL,image_dir,output_dir,datasetName,NUM_CLASSES=1,dataRatios=[0.90, 0.10, 0.0],
                                          datasetStructure=0, segExtension="mask", onlineAugmentation=1):
    
    print("================================================")
    print("    Segmentation Training Using U-Net Torch     ")
    print("================================================")
    
    logPath        = os.path.join(output_dir,datasetName+"_"+str(NUM_CLASSES)+"_log.csv")
    finalModelPath = os.path.join(output_dir,datasetName+"_"+str(NUM_CLASSES)+"_model_pth.tar")
    
    train_transforms, val_transforms = getImageTransforms(IMAGE_HEIGHT,IMAGE_WIDTH,onlineAugmentation)

    model = PNUNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    
    
    # for multiple classes, we need only to change outchannels to number of classes
    # and use Crossentropy loss
    #loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = BinaryDiceLoss()
    if NUM_CLASSES>2:
        loss_fn = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    imgLsts = getImagesList(image_dir, rt = dataRatios, doPrint=1,
                                 datasetStructure=0,
                                 segExtension="mask",                                 
                                 onlineAugmentation=1  )
  
    trnLst,valLst,tstLst = imgLsts

    train_loader, val_loader = get_loaders(
        image_dir,
        trnLst,
        valLst,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
        datasetStructure,
        segExtension,
        onlineAugmentation)

    if LOAD_MODEL:
        checkpoint = torch.load(finalModelPath)
        model.load_state_dict(checkpoint["state_dict"])


    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    
    # Writing to the file in a loop
    line = "epoch,trainingLoss,validationLoss,Dice,Time"
    with open(logPath, 'w') as file:
            file.write(line + '\n')
    file.close()
    for epoch in range(NUM_EPOCHS):
        sTime= time.time()
        trainingLoss = train_fn(epoch,NUM_EPOCHS, train_loader, model, optimizer, loss_fn, scaler,DEVICE )
        eTime= round(time.time()-sTime,2)
         
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        modelPath = os.path.join(output_dir,datasetName+"_"+str(NUM_CLASSES)+"_"+str(epoch)+"_model_pth.tar")
        torch.save(checkpoint, modelPath)

        # check accuracy
        valData = []

        #change the model mode to prediction
        model.eval()
        with torch.no_grad():
            validationLoss = 0.0
            i = 0 
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE).unsqueeze(1)
                preds = torch.sigmoid(model(x))                
                preds = (preds > 0.5).float()  
                dx =     x.cpu().numpy().transpose(0, 2, 3, 1)
                dy =     y.cpu().numpy().transpose(0, 2, 3, 1)
                dp = preds.cpu().numpy().transpose(0, 2, 3, 1)
                
                validationLoss = validationLoss + check_accuracy([dy,dp])
                if i < 11: 
                   valData.append([dx,dy,dp ])                   
                i=i+1   
        #change the model mode back to training
        model.train()
        validationLoss = round(validationLoss/(i+1),4)
        #validationLoss = round(check_accuracy(val_loader, model, device=DEVICE),2)
        # validationLoss = check_accuracy(valData)
        dice = round(1-validationLoss,4)
        # print some examples to a folder
        #save_predictions_as_imgs( val_loader, model, folder=output_dir, device=DEVICE )
        save_predictions_as_imgs( valData,folder=output_dir)

        line = str(epoch+1) +","+ str(trainingLoss) +","+ str(validationLoss) +","+ str(dice) +","+ str(eTime)+","
        print(line)
        with open(logPath, 'a') as file:
                file.write(line + '\n')
        file.close()
        plotLoss(logPath, doSave=1, doShow=0)

    torch.save(checkpoint, finalModelPath)

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
    dataRatios = [0.10,0.10,0.0]
    datasetStructure   = 0
    segExtension       = "mask"
    onlineAugmentation = 1
    main(LEARNING_RATE,DEVICE,BATCH_SIZE,NUM_EPOCHS,NUM_WORKERS,IMAGE_HEIGHT,IMAGE_WIDTH,
         PIN_MEMORY,LOAD_MODEL,image_dir,output_dir, NUM_CLASSES, dataRatios,
         datasetStructure, segExtension, onlineAugmentation)

