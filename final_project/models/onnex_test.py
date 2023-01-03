import torch.onnx 
import torch.nn as nn
import torch
import numpy as np
#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1,3,180,320, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "ImageClassifier.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput']) # the model's output names 
         #dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                #'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 


class F110_YOLO(torch.nn.Module):
    def __init__(self):
        super(F110_YOLO, self).__init__()
        # TODO: Change the channel depth of each layer
        self.conv1 = nn.Conv2d(3, 4, kernel_size = 4, padding = 1, stride = 2)
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(4, 48, kernel_size = 4, padding = 1, stride = 2)
        self.batchnorm2 = nn.BatchNorm2d(48)
        self.relu2 = nn.ReLU(inplace = True)

        self.conv3 = nn.Conv2d(48, 162, kernel_size = 4, padding = 1, stride = 2)
        self.batchnorm3 = nn.BatchNorm2d(162)
        self.relu3 = nn.ReLU(inplace = True)
        
        self.conv4 = nn.Conv2d(162, 256, kernel_size = 4, padding = 1, stride = 2)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace = True)

        self.conv5 = nn.Conv2d(256, 512, kernel_size = 4, padding = 1, stride = 2)
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace = True)

        self.conv6 = nn.Conv2d(512,256, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace = True)

        self.conv7 = nn.ConvTranspose2d(256, 64, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU(inplace = True)

        self.conv8 = nn.ConvTranspose2d(64, 6, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm8 = nn.BatchNorm2d(6)
        self.relu8 = nn.ReLU(inplace = True)

        self.conv9 = nn.Conv2d(6, 5, kernel_size = 1, padding = 0, stride = 1)
        self.relu9 = nn.ReLU()
    
    def forward(self, x):
        debug = 0 # change this to 1 if you want to check network dimensions
        if debug == 1: print(0, x.shape)
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        if debug == 1: print(1, x.shape)
        x = torch.relu(self.batchnorm2(self.conv2(x)))
        if debug == 1: print(2, x.shape)
        x = torch.relu(self.batchnorm3(self.conv3(x)))
        if debug == 1: print(3, x.shape)
        x = torch.relu(self.batchnorm4(self.conv4(x)))
        if debug == 1: print(4, x.shape)
        x = torch.relu(self.batchnorm5(self.conv5(x)))
        if debug == 1: print(5, x.shape)
        x = torch.relu(self.batchnorm6(self.conv6(x)))
        if debug == 1: print(6, x.shape)
        x = torch.relu(self.batchnorm7(self.conv7(x)))
        if debug == 1: print(7, x.shape)
        x = torch.relu(self.batchnorm8(self.conv8(x)))
        if debug == 1: print(8, x.shape)
        x = self.conv9(x)
        if debug == 1: print(9, x.shape)
        x = torch.cat([x[:, 0:3, :, :], torch.sigmoid(x[:, 3:5, :, :])], dim=1)

        return x

    def get_loss(self, result, truth, lambda_coord = 5, lambda_noobj = 1):
        x_loss = (result[:, 1, :, :] - truth[:, 1, :, :]) ** 2
        y_loss = (result[:, 2, :, :] - truth[:, 2, :, :]) ** 2
        w_loss = (torch.sqrt(result[:, 3, :, :]) - torch.sqrt(truth[:, 3, :, :])) ** 2
        h_loss = (torch.sqrt(result[:, 4, :, :]) - torch.sqrt(truth[:, 4, :, :])) ** 2
        class_loss_obj = truth[:, 0, :, :] * (truth[:, 0, :, :] - result[:, 0, :, :]) ** 2
        class_loss_noobj = (1 - truth[:, 0, :, :]) * lambda_noobj * (truth[:, 0, :, :] - result[:, 0, :, :]) ** 2

        total_loss = torch.sum(lambda_coord * truth[:, 0, :, :] * (x_loss + y_loss + w_loss + h_loss) + class_loss_obj + class_loss_noobj)
        
        return total_loss

if __name__ == "__main__": 

    # Let's build our model 
    #train(5) 
    #print('Finished Training') 

    # Test which classes performed well 
    #testAccuracy() 

    # Let's load the model we just created and test the accuracy per label 
    model = F110_YOLO()
    path = "model_1000.pt" 
    model.load_state_dict(torch.load(path)) 

    # Test with batch of images 
    #testBatch() 
    # Test how the classes performed 
    #testClassess() 
 
    # Conversion to ONNX 
    Convert_ONNX() 
