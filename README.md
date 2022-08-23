# Diabetic Retinopathy Grade Detection using Convolutional Neural Networks (CNNs) and Transfer Learning techniques

## Goals/Objectives: 
1) To automate the process of detecting the severity grades (grade 0 to grade 4) of Diabetic Retinopathy (DR) disease using Convolutional Neural Networks(CNN) with a better prediction than guessing.
2) Experimentation with different data pre-processing techniques, parameters and hyperparameters related to the model (Parameters: configuration variables internal to the model, Hyperparameters: explicitly specified parameters that control the training), and model architectures and understanding how it affects the final effectiveness of predictions
3) Implementing Filter, Kernel, Feature attribute, layer attribute visualization techniques for model explainability

## Dataset Used: 
Indian Diabetic Retinopathy Dataset (IDRID) - Disease Grading sub-section which consists of -
1. Original color fundus images of human retina (516 images divided into train set (413 images) and test set (103 images) - JPG Files)
2. Groundtruth Labels for Diabetic Retinopathy (Divided into train and test set - CSV File)

The dataset can be downloaded from this link: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

![5 stages of the disease](https://github.com/SohamBera16/Diabetic-Retinopathy-Grade-Detection-using-CNNs-and-Transfer-Learning-techniques/blob/main/5%20stages%20of%20DR.jpg)

## Baseline CNN Model execution :

Before running the commands, make sure to change or set the directory names as necessary in all the relevant occurances of the "dir" variable across different modules.

- Run main.py in the root directory using the command -

  `python3 main.py`
  
## Improved Model versions execution: 
- In order to run the different versions, the corresponding parameters in the main method of main.py have to be tweaked. Different options of changing parameters are mentioned as follows -

### Dataset Parameters

In the first part of the main method, the dataset parameters can be tweaked. 
- batch_size and input_size can be changed.
- Cross validation, per image normalization, per dataset normalization and data augmentation can be activated.

### Model Initialization: 

Here, the model to be used for training and making predictions is initialized.
- A deep network can be imported through the *sel* parameter. ["sel" stands for selection.]
- If *sel* is None, the baseline CNN will be initialized.

### Network Parameters

The parameters of the network can be tuned by accordingly changing the following variables -
- The learning rate and number of epochs can be changed.
- The current optimizer is ADAM.
- The current loss function is the Cross Entropy Loss

### General Parameters

- The *save_name* acts as an identifier and is very important. Two directories will be created according to *save_name*. The checkpoints and logs will be stored in these folders. If a *save_name* is used a second time, the results in the corresponding folders will be overwritten.
- The *operation* parameter can be used to change the type of operation. Currently 2 is selected which will train and test the model.

Once you are satisfied with the settings, simply run the main.py from the root directory:

  `python3 main.py`
  
### Results

By using the Alexnet and Resnet18 models with pretrained weights

## Disclaimer: 
This project was developed by the author as part of the Fachpraktikum (advanced internship): Artificial Intelligence within the University of Stuttgart. [Year: 2022]
