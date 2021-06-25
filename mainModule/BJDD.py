from os import write
import pdb
import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import sys
import glob
import time
import colorama
from colorama import Fore, Style
from etaprogress.progress import ProgressBar
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
from utilities.torchUtils import *
from dataTools.customDataloader import *
from utilities.inferenceUtils import *
from utilities.aestheticUtils import *
from loss.pytorch_msssim import *
from loss.colorLoss import *
from loss.percetualLoss import *
from modelDefinitions.attentionDis import *
from modelDefinitions.attentionGen import *
from torchvision.utils import save_image
from tqdm import tqdm


class BJDD:
    def __init__(self, config):
        
        # Model Configration 
        self.gtPath = config['gtPath']
        self.targetPath = config['targetPath']
        self.checkpointPath = config['checkpointPath']
        self.logPath = config['logPath']
        self.testImagesPath = config['testImagePath']
        self.custom_test_set = config['custom_test_md_set']
        self.resultDir = config['resultDir']
        self.modelName = config['modelName']
        self.dataSamples = config['dataSamples']
        self.batchSize = int(config['batchSize'])
        self.imageH = 128#int(config['imageH'])
        self.imageW = 128#int(config['imageW'])
        self.inputC = int(config['inputC'])
        self.outputC = int(config['outputC'])
        self.scalingFactor = int(config['scalingFactor'])
        self.binnigFactor = int(config['binnigFactor'])
        self.totalEpoch = int(config['epoch'])
        self.interval = int(config['interval'])
        self.learningRate = float(config['learningRate'])
        self.adamBeta1 = float(config['adamBeta1'])
        self.adamBeta2 = float(config['adamBeta2'])
        self.barLen = int(config['barLen'])
        
        # Initiating Training Parameters(for step)
        self.currentEpoch = 0
        self.startSteps = 0
        self.totalSteps = 0
        self.adversarialMean = 0
        self.PR = 0.0

        # Normalization
        self.unNorm = UnNormalize()

        # Noise Level for inferencing
        self.noiseSet = [0, 5, 10, 15]
        

        # Preapring model(s) for GPU acceleration
        self.device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.attentionNet = attentionNet().to(self.device)
        self.discriminator = attentiomDiscriminator().to(self.device)

        # Optimizers
        self.optimizerEG = torch.optim.Adam(self.attentionNet.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        self.optimizerED = torch.optim.Adam(self.discriminator.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        
        # Scheduler for Super Convergance
        self.scheduleLR = None
        
    def customTrainLoader(self, overFitTest = False):
        
        targetImageList = imageList(self.targetPath)
        print ("Trining Samples (Input):", self.targetPath, len(targetImageList))

        if overFitTest == True:
            targetImageList = targetImageList[-1:]
        if self.dataSamples:
            targetImageList = targetImageList[:self.dataSamples]

        datasetReadder = customDatasetReader(   
                                                image_list=targetImageList, 
                                                imagePathGT=self.gtPath,
                                                height = self.imageH,
                                                width = self.imageW,
                                            )

        self.trainLoader = torch.utils.data.DataLoader( dataset=datasetReadder,
                                                        batch_size=self.batchSize, 
                                                        shuffle=True
                                                        )
        
        return self.trainLoader

    def modelTraining(self, resumeTraning=False, overFitTest=False, dataSamples = None):
        
        if dataSamples:
            self.dataSamples = dataSamples 

        # Losses
        featureLoss = regularizedFeatureLoss().to(self.device)
        reconstructionLoss = torch.nn.L1Loss().to(self.device)
        ssimLoss = MSSSIM().to(self.device)
        colorLoss = deltaEColorLoss(normalize=True).to(self.device)
        adversarialLoss = nn.BCELoss().to(self.device)
 
        # Overfitting Testing
        if overFitTest == True:
            customPrint(Fore.RED + "Over Fitting Testing with an arbitary image!", self.barLen)
            trainingImageLoader = self.customTrainLoader(overFitTest=True)
            self.interval = 1
            self.totalEpoch = 100000
        else:  
            trainingImageLoader = self.customTrainLoader()


        # Resuming Training
        if resumeTraning == True:
            self.modelLoad()
            try:
                pass#self.modelLoad()

            except:
                #print()
                customPrint(Fore.RED + "Would you like to start training from sketch (default: Y): ", textWidth=self.barLen)
                userInput = input() or "Y"
                if not (userInput == "Y" or userInput == "y"):
                    exit()
        

        # Starting Training
        customPrint('Training is about to begin using:' + Fore.YELLOW + '[{}]'.format(self.device).upper(), textWidth=self.barLen)
        
        # Initiating steps
        self.totalSteps =  int(len(trainingImageLoader)*self.totalEpoch)
        startTime = time.time()
        interv_time = startTime
        # Instantiating Super Convergance 
        #self.scheduleLR = optim.lr_scheduler.OneCycleLR(optimizer=self.optimizerEG, max_lr=self.learningRate, total_steps=self.totalSteps)
        # Initiating progress bar 
        bar = ProgressBar(self.totalSteps, max_width=int(self.barLen/2))
        # import pdb; pdb.set_trace()
        currentStep = self.startSteps

        # utilize tensorboard
        createDir(self.logPath)
        writer = SummaryWriter(self.logPath + f"training_process_{self.modelName}")

        while currentStep < self.totalSteps:

            # Time tracker
            iterTime = time.time()
            for LRImages, HRGTImages in trainingImageLoader:
                
                ##############################
                #### Initiating Variables ####
                ##############################
                # Updating Steps
                if currentStep > self.totalSteps:
                    self.savingWeights(currentStep)
                    customPrint(Fore.YELLOW + "Training Completed Successfully!", textWidth=self.barLen)
                    exit()
                currentStep += 1

                # Images
                rawInput = LRImages.to(self.device)
                highResReal = HRGTImages.to(self.device)
              
                
                # GAN Variables
                # TODO 
                onesConst = torch.ones(rawInput.shape[0], 1).to(self.device)
                targetReal = (torch.rand(rawInput.shape[0],1) * 0.5 + 0.7).to(self.device)
                targetFake = (torch.rand(rawInput.shape[0],1) * 0.3).to(self.device)


                ##############################
                ####### Training Phase #######
                ##############################
    
                # Image Generation
                highResFake = self.attentionNet(rawInput)
                
                # Optimaztion of Discriminator
                self.optimizerED.zero_grad()
                ed_true_loss = adversarialLoss(self.discriminator(highResReal), targetReal)
                ed_false_loss = adversarialLoss(self.discriminator(highResFake.detach()), targetFake)
                lossED = ed_true_loss + ed_false_loss
                lossED.backward()
                self.optimizerED.step()

                
                # Optimization of generator 
                self.optimizerEG.zero_grad()
                feature_loss = featureLoss(highResFake, highResReal)
                reconstruction_loss = reconstructionLoss(highResFake, highResReal)
                color_loss = colorLoss(highResFake, highResReal)
                generatorContentLoss =  feature_loss + reconstruction_loss + color_loss

                generatorAdversarialLoss = adversarialLoss(self.discriminator(highResFake), onesConst)
                lossEG = generatorContentLoss + 1e-3 * generatorAdversarialLoss
                lossEG.backward()
                self.optimizerEG.step()

                # Steps for Super Convergance            
                #self.scheduleLR.step()

                ##########################
                ###### Model Logger ######
                ##########################   

                # Progress Bar
                if (currentStep  + 1) % self.interval/2 == 0:
                    # pdb.set_trace()
                    bar.numerator = currentStep + 1
                    print(Fore.YELLOW + "Steps |",bar,Fore.YELLOW + "| LossEG: {:.4f}, LossED: {:.4f}, RFL: {:.4f}, run_Time: {}".format(lossEG, lossED, feature_loss, (time.time()-interv_time)),end='\r')
                    interv_time = time.time()
                
                # Updating training log
                if (currentStep + 1) % self.interval == 0:
                    # record G_losses
                    writer.add_scalar("LossEG", lossEG, global_step=currentStep+1)
                    writer.add_scalar("RFL", feature_loss, global_step=currentStep+1)
                    writer.add_scalar("Color_loss", color_loss, global_step=currentStep+1)
                    writer.add_scalar("L1_loss", reconstruction_loss, global_step=currentStep+1)

                    # record loss_D
                    writer.add_scalar("LossED", lossED, global_step=currentStep+1)
                    writer.add_scalar("D_real_loss", ed_true_loss, currentStep+1)
                    writer.add_scalar("D_fake_loss", ed_false_loss, currentStep+1)

                    writer.add_image("Input_Images", torchvision.utils.make_grid(self.unNorm(rawInput[:8])), global_step=currentStep+1)
                    writer.add_image("Generated_images", torchvision.utils.make_grid(self.unNorm(highResFake[:8])), global_step=currentStep + 1)
                    writer.add_image("GT_image", torchvision.utils.make_grid(self.unNorm(highResReal[:8])), global_step=currentStep+1)
                    # Updating Tensorboard
                    # summaryInfo = { 
                    #                 'Input Images' : self.unNorm(rawInput),
                    #                 'AttentionNetGen Images' : self.unNorm(highResFake),
                    #                 'GT Images' : self.unNorm(highResReal),
                    #                 'Step' : currentStep + 1,
                    #                 'Epoch' : self.currentEpoch,
                    #                 'LossEG' : lossEG.item(),
                    #                 'LossED' : lossED.item(),
                    #                 'Path' : self.logPath,
                    #                 'Atttention Net' : self.attentionNet,
                    #               }
                    # tbLogWritter(summaryInfo)
                    save_image(self.unNorm(highResFake[0]), 'modelOutput.png')

                    # Saving Weights and state of the model for resume training 
                    self.savingWeights(currentStep)
                
                if (currentStep + 1) % (self.interval ** 2) == 0 : 
                    print("\n")
                    self.savingWeights(currentStep + 1, True)
                    #self.modelInference(validation=True, steps = currentStep + 1)
                    eHours, eMinutes, eSeconds = timer(iterTime, time.time())
                    print (Fore.CYAN +'Steps [{}/{}] | Time elapsed [{:0>2}:{:0>2}:{:0>2}] | LossC: {:.2f}, LossP : {:.2f}, LossEG: {:.2f}, LossED: {:.2f}' 
                            .format(currentStep + 1, self.totalSteps, eHours, eMinutes, eSeconds, color_loss, feature_loss, lossEG, lossED))
        
        writer.close()
        print(f"total running time: {time.time() - startTime}")           
   
    def modelInference(self, testImagesPath = None, outputDir = None, resize = None, validation = None, noiseSet = None, steps = None):
        if not validation:
            self.modelLoad()
            print("\nInferencing on pretrained weights.")
        else:
            print("Validation about to begin.")
        if not noiseSet:
            noiseSet = self.noiseSet
        if testImagesPath:
            self.testImagesPath = testImagesPath
        if outputDir:
            self.resultDir = outputDir
        

        modelInference = inference(gridSize=self.binnigFactor, inputRootDir=self.testImagesPath, outputRootDir=self.resultDir, modelName=self.modelName, validation=validation)

        testImageList = modelInference.testingSetProcessor()
        barVal = ProgressBar(len(testImageList) * len(noiseSet), max_width=int(50))
        imageCounter = 0
        with torch.no_grad():
            for noise in noiseSet:
                #print(noise)
                for imgPath in testImageList:
                    img = modelInference.inputForInference(imgPath, noiseLevel=noise).to(self.device)
                    output = self.attentionNet(img)
                    modelInference.saveModelOutput(output, imgPath, noise, steps)
                    imageCounter += 1
                    if imageCounter % 2 == 0:
                        barVal.numerator = imageCounter
                        print(Fore.CYAN + "Image Processd |", barVal,Fore.CYAN, end='\r')
        print("\n")
    
    def model_custom_inference(self, testImagesPath = None, outputDir = None, validation = None, steps = None):
        test_set = self.custom_test_set
        if not validation:
            self.modelLoad()
            print("\nInferencing on pretrained weights.")
        else:
            print("Validation about to begin.")
        modelInference = inference(gridSize=self.binnigFactor, inputRootDir=self.custom_test_set, outputRootDir=self.resultDir, modelName=self.modelName, validation=validation)

        with torch.no_grad():
            # traverse image folder
            for root, _, files in os.walk(test_set):
                for name in tqdm(files):
                    img = modelInference.custom_inputForInference(os.path.join(root, name)).to(self.device)
                    output = self.attentionNet(img)
                    noise = int(name[name.find('_')+1:name.find('.')])
                    modelInference.saveModelOutput(output, os.path.join(root, name), noise, steps)
        print("finished")


    def modelSummary(self,input_size = None):
        if not input_size:
            input_size = (3, self.imageH//self.scalingFactor, self.imageW//self.scalingFactor)

     
        customPrint(Fore.YELLOW + "AttentionNet (Generator)", textWidth=self.barLen)
        summary(self.attentionNet, input_size =input_size)
        print ("*" * self.barLen)
        print()

        customPrint(Fore.YELLOW + "AttentionNet (Discriminator)", textWidth=self.barLen)
        summary(self.discriminator, input_size =input_size)
        print ("*" * self.barLen)
        print()

        '''flops, params = get_model_complexity_info(self.attentionNet, input_size, as_strings=True, print_per_layer_stat=False)
        customPrint('Computational complexity (Enhace-Gen):{}'.format(flops), self.barLen, '-')
        customPrint('Number of parameters (Enhace-Gen):{}'.format(params), self.barLen, '-')

        flops, params = get_model_complexity_info(self.discriminator, input_size, as_strings=True, print_per_layer_stat=False)
        customPrint('Computational complexity (Enhace-Dis):{}'.format(flops), self.barLen, '-')
        customPrint('Number of parameters (Enhace-Dis):{}'.format(params), self.barLen, '-')
        print()'''

        configShower()
        print ("*" * self.barLen)
    
    def savingWeights(self, currentStep, duplicate=None):
        # Saving weights 
        checkpoint = { 
                        'step' : currentStep + 1,
                        'stateDictEG': self.attentionNet.state_dict(),
                        'stateDictED': self.discriminator.state_dict(),
                        'optimizerEG': self.optimizerEG.state_dict(),
                        'optimizerED': self.optimizerED.state_dict(),
                        'schedulerLR': self.scheduleLR
                        }
        saveCheckpoint(modelStates = checkpoint, path = self.checkpointPath, modelName = self.modelName)
        if duplicate:
            saveCheckpoint(modelStates = checkpoint, path = self.checkpointPath + str(currentStep) + "/",currentEpoch=currentStep , modelName = self.modelName, backup=None)



    def modelLoad(self):

        customPrint(Fore.RED + "Loading pretrained weight", textWidth=self.barLen)

        previousWeight = loadCheckpoints(self.checkpointPath, self.modelName)
        self.attentionNet.load_state_dict(previousWeight['stateDictEG'])
        self.discriminator.load_state_dict(previousWeight['stateDictED'])
        self.optimizerEG.load_state_dict(previousWeight['optimizerEG']) 
        self.optimizerED.load_state_dict(previousWeight['optimizerED']) 
        self.scheduleLR = previousWeight['schedulerLR']
        self.startSteps = int(previousWeight['step'])
        # pdb.set_trace()
        customPrint(Fore.YELLOW + "Weight loaded successfully", textWidth=self.barLen)


