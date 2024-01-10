import os, sys, cv2, torch, time, shutil

unetLibPath    =os.path.expanduser("~/myGit/DNNs/imageSeg/Persson/pnUnet")
iaUtilsLibPath =os.path.expanduser("~/myGit/DNNs/imageSeg/iaUtils")

DataSetName  = "Carvana"
datasetPath  = "/mnt/tnasData/dnnData/Carvana" # CarvanaDataset
image_dir    = os.path.join(datasetPath,"inputData")
output_dir   = os.path.join(datasetPath,"outputData")   

NUM_EPOCHS    = 2
NUM_WORKERS   = 1
BATCH_SIZE    = 16
LEARNING_RATE = 1e-4
# IMAGE_HEIGHT  = 160  # 1280 originally 480,320
# IMAGE_WIDTH   = 240  # 1918 originally
IMAGE_HEIGHT  = 320  # 1280 originally
IMAGE_WIDTH   = 480  # 1918 originally
PIN_MEMORY    = True
LOAD_MODEL    = False
NUM_CLASSES = 1
dataRatios=[0.10, 0.05, 0.85]
#Reset files:
# one may need to backup the folder
if os.path.exists(output_dir):
    # Remove the directory and all its contents
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)

# add our repository to python path
sys.path.append(unetLibPath)
sys.path.append(iaUtilsLibPath)

doTorchTraining = 1
if doTorchTraining:
    
    DEVICE        = "cuda" if torch.cuda.is_available() else "CPU"
    print("DEVICE Torch: ", DEVICE)
       
    import ptTrain, utils
        
    # check class size:
    doCheckClassSize = 0
    if doCheckClassSize:
        imgLsts = utils.getImagesList(image_dir, rt = dataRatios, doPrint=1)
        segLst = [x[1] for x in imgLsts[0]]
        cB = 0; cW=0
        for x in segLst:
            imgPath = os.path.join(image_dir,x)
            classSize = utils.getClassSize(imgPath)
            cB = cB +classSize[0][1]
            cW = cW +classSize[1][1]
        # It seems the network works even with large background up to 80%
        print(round(cB/len(segLst),2), round(cW/len(segLst),2))
        # 0.79 0.21
    
    sTime = time.time()
    ptTrain.main(LEARNING_RATE, DEVICE, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, IMAGE_HEIGHT,IMAGE_WIDTH,PIN_MEMORY,LOAD_MODEL,
                 image_dir,output_dir,DataSetName,NUM_CLASSES,dataRatios)
    print("training time: ", time.time()-sTime, " seconds")

doTFTraining = 0

if doTFTraining:
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    
    DEVICE = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    print("DEVICE TensorFlow: ", DEVICE)
    
    import tfUtils, tfTrain
       
    # check class size:
    doCheckClassSize = 0
    if doCheckClassSize:
        imgLsts = utils.getImagesList(image_dir, rt = [0.70, 0.10, 0.20], doPrint=1)
        segLst = [x[1] for x in imgLsts[0]]
        cB = 0; cW=0
        for x in segLst:
            imgPath = os.path.join(image_dir,x)
            classSize = utils.getClassSize(imgPath)
            cB = cB +classSize[0][1]
            cW = cW +classSize[1][1]
        # It seems the network works even with large background up to 80%
        print(round(cB/len(segLst),2), round(cW/len(segLst),2))
        # 0.79 0.21
    
    sTime = time.time()
    train.main(LEARNING_RATE, DEVICE, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, IMAGE_HEIGHT,IMAGE_WIDTH,PIN_MEMORY,LOAD_MODEL,
               image_dir,output_dir,DataSetName,NUM_CLASSES,dataRatios)
    print("training time: ", time.time()-sTime, " seconds")