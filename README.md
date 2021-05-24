# Dog Breed Identification


## Project Overview
The project demonstrates on how transfer learning can be applied to a given problem to improve accuracy and efficieny of the models. The project utilises various concepts to prove that with limited time and resources, models can be developed. The models utilised use Imagenet dataset to train and identify the categories present in the data. We couple the trained weights with our own model as an input to further identify dog breeds.

Additionally, a web application is developed that makes use of trained models. Provided an image as input identifies if a human or dog is present. If Dog is present, it identifies the dog's breed. If human is present, it identifies what dog breed does the human resemble.


## Files
**dog_app.ipynb** contains a jupyter notebook demostrating data analysis and model development
**webapp** folder has a developed simple web application for this project


## Installation
Install required libraries using requirements file using the following instruction.
```
pip install -r requirements.txt 
```


## Instructions to run web application
1. go to Webapp's directory
```
cd webapplication
```
2. Run following command to start webapp
```
python app.py

```
By default, the app runs at http://0.0.0.0:3001/


## Dataset
Dog images to train model can be found at following:
[link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

Dataset contains dog images from 133 different breeds. Train and Validation sets are used for training and fine tuning the  model while for evaluation, Test set is used.
Below is an overview of dataset.
* There are 133 total dog categories.
* There are 8351 total dog images.
* There are 6680 training dog images.
* There are 835 validation dog images.
* There are 836 test dog images.

### Metrics and Model Evaluation
**Accuracy** on testset is used as evaluation metric. Test set is used for final model evaluation.

## Proposition
* Step 1: Detect Human
* Step 2: Detect Dog
* Step 3: Compare preformance of various CNN architectures to classify dog breeds 
    * a: CNN From Scratch
    * b: Transfer learning using VGG16
    * c: Transfer lerning using InceptionV3
* Step 4: Final prediction
* Findings

### Step 1: Detect Human
OpenCV's implementation of:
[Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) 

OpenCV provides many pre-trained face detectors, stored as XML files on github. Haarcascades directory contains a pre-trained face detector.
If a human image is provided to the module, it returns the coordinates around the detacted face.
So this opencv classifier is used only to detect humans and for dog detection, another classifier is used.

### Step 2: Detect Dog
A pre-trained ResNet-50 model is used:
[ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) 

The model is trained on [ImageNet](http://www.image-net.org/) containing over 10 million URLs, each linking to an image containing object from one of 1000 categories. Given an image, this pre-trained ResNet-50 model returns a prediction for the object that is contained in the image.
This model was able to detect dogs in images with an accuracy of 100%.

We could have used the above mentioned ResNet based model for detecting the human presense but the ImageNet dataset does not explicitly cover human as a category.

### Step 3: Compare preformance of different CNNs architectures to dlassify Dog Breeds 
Now that we have identified that if a dog or human is present in the image or not, next step is to identify its breed. CNNs are used for this purpose.
Following are the experimentations performed:
#### a: CNN From Scratch
A CNN model is build from scratch with the following architecture.
A model having 4 convolutional layers following maxpool layers was trained for 20 epochs. 
Model achieved an accuracy of above 10%.
#### b: Transfer learning using VGG16
To reduce training time without sacrificing accuracy, transfer learning is used. VGG16 is used as pretrained model to obtain bottleneck features of each input image. A new model is created and trained consisting of GlobalAveragePooling layer and final Dense layer to predict 133 dog breeds. 
Model achieved an accuracy of above 40%.
#### c: Transfer lerning using InceptionV3
InceptionV3 is also used as pretrained model to calculate bottleneck featues which is then fed to a second model consisting of GlobalAveragePooling layer following two fully connected dense layers with dropout to predict 133 dog breeds.
Model achieved an accuracy of above 80%.
#### Step 4: Final prediction
The final function that is used to predict dog breeds works in following way:
1. Detects if the provided image is either a human, dog or has an absence of both of these
2. If dog is detected, a predicted breed is returned
3. If human is detected, a predicted breed is returned resembling that human 
4. If human and dog are not detected, a relevan message is returned

#### Conclusion
* Transfer learning is very useful of the datasets have similarities among them
* It greatly reduces the time and resources required to train a complex model


## Application Dashboard
App main page:
![Main page](https://github.com/rafayullah/Udacity-DogBreedClassifier/blob/main/images/Dashboard_Image.png?raw=true)


## Authors
* [Rafay Ullah Choudhary](https://github.com/rafayullah)
