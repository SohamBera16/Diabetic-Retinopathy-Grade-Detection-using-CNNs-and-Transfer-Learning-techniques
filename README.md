## Facilitating the diagnosis of Diabetic Retinopathy through Deep Learning

## Authors: Soham Kanti Bera and Sinan Kurtyigit

Objectives: 1) To automate the process of detecting Diabetic Retinopathy (DR) disease severity grades (grade 0 to grade 4) using Convolutional Neural Networks(CNN) with a better prediction than guessing.
2) Experimentation with different parameters, data pre-processing techniques, and model architectures and study how it affects the final performance of the model
3) Implementing Filter, Kernel, Feature attribute, layer attribute visualization techniques for model explainability

## Baseline
- Run main.py in the root directory (AC-DL-LAB-SS-2022-Team02)

  `python3 main.py`
  
## Improved Versions
- In order to run the different versions, the corresponding parameters in the main method of main.py have to be tweaked.
### Dataset Parameters
In the first part of the main method, the dataset parameters can be tweaked. 
- batch_size and input_size can be changed.
- Cross validation, per image normalization, per dataset normalization and data augmentation can be activated.

### Init Network
Here, the network is initialized
- A deep network can be imported through the *sel* parameter.
- If *sel* is None, the baseline CNN will be initialized.

### Network Parameters
Here the parameters of the network can be tuned
- The learning rate and number of epochs can be changed.
- The current optimizer is ADAM.
- The current loss function is the Cross Entropy Loss

### General Parameters
Next are some general parameters
- The *save_name* acts as an identifier and is very important. Two directories will be created according to *save_name*. The checkpoints and logs will be stored in these folders. If a *save_name* is used a second time, the results in the corresponding folders will be overwritten.
- The operation parameter can be used to change the type of operation. Currently 2 is selected which will train and test the model.

Once you are satisfied with the settings, simply run the main.py from the root directory (AC-DL-LAB-SS-2022-Team02):

  `python3 main.py`

