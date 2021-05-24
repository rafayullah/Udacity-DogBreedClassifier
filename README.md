# Dog Breed Identification


## Project Definition
The project demonstrates how transfer learning can be applied to a given problem to improve the accuracy and efficiency of the models. The project utilizes various concepts to prove that with limited time and resources, models can be developed. The models utilized use the Imagenet dataset to train and identify the categories present in the data. We couple the trained weights with our own model as an input to further identify dog breeds.

Additionally, a web application is developed that makes use of trained models. Provided an image as input identifies if a human or dog is present. If Dog is present, it identifies the dog's breed. If a human is present, it identifies what dog breed does the human resemble.


## Files
**dog_app.ipynb** contains a jupyter notebook demonstrating data analysis and model development
**webapplication** folder has a developed simple web application for this project


## Installation
Install required libraries using the requirements file using the following instruction.
```
pip install -r requirements.txt 
```


## Instructions to run the web application
1. go to Webapp's directory
```
cd webapplication
```
2. Run the following command to start webapp
```
python app.py

```
By default, the app runs at http://0.0.0.0:3001/


## Problem Statement
### Identification of dog breeds
### Description: 
The project targets to deliver an analysis of various approaches to develop models for the classification of various dog breeds present in an input image. Additionally, due to the integration of transfer learning, the model is expected to have efficient and accurate classification using limited resources and training time.
At project will also include a web application for the purpose of demonstration.

## Dataset
Dog images to train model can be found at following:
[link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

The dataset contains dog images from 133 different breeds. Train and Validation sets are used for training and fine-tuning the model while for evaluation, a Test set is used.
Below is an overview of the dataset.
* There are 133 total dog categories.
* There are 8351 total dog images.
* There are 6680 training dog images.
* There are 835 validation dog images.
* There are 836 test dog images.

### Metrics and Model Evaluation
**Accuracy** on the test set is used as an evaluation metric. The test set is used for the final model evaluation.

## Proposition
* Step 1: Detect Human
* Step 2: Detect Dog
* Step 3: Compare performance of various CNN architectures to classify dog breeds 
    * a: CNN From Scratch
    * b: Transfer learning using VGG16
    * c: Transfer learning using InceptionV3
* Step 4: Final prediction
* Findings

### Step 1: Detect Human
OpenCV's implementation of:
[Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) 

OpenCV provides many pre-trained face detectors, stored as XML files on GitHub. Haarcascades directory contains a pre-trained face detector.
If a human image is provided to the module, it returns the coordinates around the detected face.
So this OpenCV classifier is used only to detect humans and for dog detection, another classifier is used.

### Step 2: Detect Dog
A pre-trained ResNet-50 model is used:
[ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) 

The model is trained on [ImageNet](http://www.image-net.org/) containing over 10 million URLs, each linking to an image containing an object from one of 1000 categories. Given an image, this pre-trained ResNet-50 model returns a prediction for the object that is contained in the image.
This model was able to detect dogs in images with an accuracy of 100%.

We could have used the above-mentioned ResNet based model for detecting the human presence but the ImageNet dataset does not explicitly cover humans as a category.

### Step 3: Compare performance of different CNNs architectures to classify Dog Breeds 
Now that we have identified that if a dog or human is present in the image or not, the next step is to identify its breed. CNN's are used for this purpose.
Following are the experimentations performed:
#### a: CNN From Scratch
A CNN model is built from scratch with the following architecture.
A model having 4 convolutional layers following maxpool layers was trained for 20 epochs. 
The model achieved an accuracy of above 10%.
#### b: Transfer learning using VGG16
To reduce training time without sacrificing accuracy, transfer learning is used. VGG16 is used as a pre-trained model to obtain bottleneck features of each input image. A new model is created and trained to consist of GlobalAveragePooling layer and the final Dense layer to predict 133 dog breeds. 
The model achieved an accuracy of above 40%.
#### c: Transfer learning using InceptionV3
InceptionV3 is also used as a pre-trained model to calculate bottleneck features which are then fed to a second model consisting of GlobalAveragePooling layer following two fully connected dense layers with dropout to predict 133 dog breeds.
The model achieved an accuracy of above 80%.
### Step 4: Final prediction
The final function that is used to predict dog breeds works in the following way:
1. Detects if the provided image is either a human, dog or has an absence of both of these
2. If a dog is detected, a predicted breed is returned
3. If a human is detected, a predicted breed is returned resembling that human 
4. If human and dog are not detected, a relevant message is returned

#### Conclusion
* Transfer learning is very useful for the datasets having similarities among them
* It greatly reduces the time and resources required to train a complex model


## Application Dashboard
App main page:
![Main page](https://github.com/rafayullah/Udacity-DogBreedClassifier/blob/main/images/Dashboard_Image.png?raw=true)


## Authors
* [Rafay Ullah Choudhary](https://github.com/rafayullah)
