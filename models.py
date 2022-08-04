import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, inputNode=561, hiddenNode=256, outputNode=5):
        super(FC, self).__init__()
        # Define Hyperparameters
        self.inputLayerSize = inputNode
        self.outputLayerSize = outputNode
        self.hiddenLayerSize = hiddenNode

        # weights
        self.Linear1 = nn.Linear(self.inputLayerSize, self.hiddenLayerSize)
        self.Linear2 = nn.Linear(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        if len(X.shape) > 3:  # received 2D input -> squeeze
            input = X.view(X.size(0), -1)
        else:
            input = X
        # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.Linear1(input)
        self.a2 = self.sigmoid(self.z2)  # activation function
        self.z3 = self.Linear2(self.a2)
        return self.z3

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+torch.exp(-z))

    def loss(self, yHat, y):
        J = 0.5*sum((y-yHat)**2)


class CNN(nn.Module):
    def __init__(self, num_classes=5, applySigmoid=False):
        super(CNN, self).__init__()

        self.conv11 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv12 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(16 * 16 * 64, num_classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.sigmoid = nn.Sigmoid()
        self.applySigmoid = applySigmoid

        self.relu = nn.ReLU()   # activation function

    def forward(self, x):
        out11 = self.maxpool(self.relu(self.conv11(x)))

        out12 = self.maxpool(self.relu(self.conv12(out11)))

        out = out12.reshape(out12.size(0), -1)
        out = self.fc(out)

        if self.applySigmoid:
            out = self.sigmoid(out)

        return out


# This is the baseline CNN model against which we will evaluate our improved model's success
class CNNBase(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNBase, self).__init__()

        self.conv11 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv12 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(16 * 16 * 64, num_classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        out11 = self.maxpool(self.relu(self.conv11(x)))

        # print(out11.shape)
        out12 = self.maxpool(self.relu(self.conv12(out11)))

        # print(out12.shape)

        out = out12.reshape(out12.size(0), -1)
        out = self.fc(out)

        return out


class EnsembleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(EnsembleCNN, self).__init__()

        self.cnn1 = CNN(num_classes)
        self.cnn2 = CNN(num_classes)
        self.cnn3 = CNN(num_classes)

    def forward(self, x):
        outCNN1 = self.cnn1(x)
        outCNN2 = self.cnn2(x)
        outCNN3 = self.cnn3(x)

        out = torch.divide(outCNN1+outCNN2+outCNN3, 3)

        return out


# This function allows us to import deep networks like AlexNet and ResNet.
def get_pretrained_models(model_name='resnet', num_classes=5, freeze_prior=True, use_pretrained=True):

    from torchvision import models
    from utils import set_parameter_requires_grad

    # Initialize these variables which will be set in this if statement. 
    # Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
