# Usefull common functions

import os, sys, time, shutil, csv, json, cv2
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from PIL import Image


# Change image format e.g. from gif to png
# requires pip install pillow
def imageReformat(inputFolderPath, inputFormat=None,  outputFormat="png", outputFolderPath=None):
    # inputFolderPath could be image path or a folder path contains multiple images
    if os.path.isfile(inputFolderPath):
        print(inputFolderPath)
        # Split the path and the filename
        file_path, filename = os.path.split(inputFolderPath)

        # Split the filename and extension
        filename, inputFormat = os.path.splitext(filename)
        # Open the original image
        img = Image.open(inputFolderPath)

        # Save the image in a new format
        if not outputFolderPath is None:
           if os.path.isfile(outputFolderPath):
              img.save(outputFolderPath)
           else:
              img.save(os.path.join(outputFolderPath,filename+"."+outputFormat))
        else:
            # save in the same folder      
            img.save(os.path.join(file_path,filename+"."+outputFormat))
    else:
        # change format of all files in the folder
        fnms = os.listdir(inputFolderPath)
        fnms = [x for x in fnms if inputFormat in x ]
        for fnm in fnms:
            print(imgPath)
            imgPath = os.path.join(inputFolderPath,fnm)
            img = Image.open(imgPath)

            file_path, filename = os.path.split(imgPath)
            filename, inputFormat = os.path.splitext(filename)
            
            if outputFolderPath is None:
               # Save the image in the same folder
               img.save(os.path.join(file_path,filename+"."+outputFormat))
            else:
               img.save(os.path.join(outputFolderPath,filename+"."+outputFormat)) 
    print("imageReformat done! ..................")            
    # Note: in Linux one can use the terminal e.g. in Ubuntu: 
    # sudo apt install imagemagick
    # cmd = 'for file in '+inputPath+'/*.gif; do convert "$file" '+outputPath+'"/${file%.*}.png" done'
    # os.system(cmd)


import os, time, cv2
from collections import Counter
from PIL import Image
import numpy as np

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


def check_accuracy(valData):
    dice_score  = 0
    for x, y, p in valData:
        dice = (2 * (p * y).sum()) / ((p + y).sum() + 1e-8 )
        dice_score += dice
    dice_score = dice_score/len(valData)
    validationLoss = round(1.0- dice_score,2)
    return validationLoss


def mergeSaveImages(x, filePath,doSave=1):
   # Determine the total width and maximum height
    total_width = sum(image.shape[1] for image in x)
    max_height  = x.shape[1]
    # Create a new blank image with total width and maximum height
    composite_image = Image.new('RGB', (total_width, max_height))

    # Paste each image into the composite image
    x_offset = 0
    for image in x:        
         # Handle single-channel images by squeezing the last dimension if it is 1
        if image.shape[-1] == 1:
            image = image.squeeze(-1)

        img = Image.fromarray(np.uint8(image * 255))  # Scale to 0-255 if necessary
        composite_image.paste(img, (x_offset, 0))
        x_offset += img.width

    if doSave:
        # Save the composite image
        composite_image.save(filePath)    
    return composite_image

def save_predictions_as_imgs(data, folder="saved_images/"):
    for idx, (x, y, p) in enumerate(data):
        img = mergeSaveImages(x, os.path.join(folder, "img_" + str(idx) + "_img.png"),0)
        seg = mergeSaveImages(y, os.path.join(folder, "img_" + str(idx) + "_seg.png"),0)
        prd = mergeSaveImages(p, os.path.join(folder, "img_" + str(idx) + "_prd.png"),0)
        blended_image = Image.blend(img, prd, alpha=0.5)
        blended_image.save(os.path.join(folder, "img_" + str(idx) + "_img-prd.png"))    

def plotLoss(csvPath, doSave=1, doShow=0):

    #Read the CSV file
    df = pd.read_csv(csvPath,  index_col=False) 
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['trainingLoss'],   label='Training',   color='blue')
    plt.plot(df['epoch'], df['validationLoss'], label='Validation', color='red')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training and Validation Losses per Epochs')
    plt.legend()

    if doSave: 
       file_path, filename = os.path.split(csvPath)
       outputPath =os.path.join(file_path,filename[:-7]+"result.png") 
       plt.savefig(outputPath)  # Saves the plot as 'plot.png'
    if doShow:
       plt.show()  # Optionally display the plot