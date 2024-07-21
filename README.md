# Birds object detection Part II

## Introduction
This codebase includes the code required to label your images, build your data model and look at the results.

## Setting up the environment
### Clone the codebase
```
cd /home/pi
git clone https://github.com/jeffspiinthesky/UK-Bird-Classifier-Pt-2.git
```

### Install pyenv
```
curl https://pyenv.run | bash
```
As instructed, add extra config into /home/pi/.bashrc as required

### Install python 3.9
NOTE: Tensorflow doesn't seem to like python 3.10 (at time of writing anyway) so this required python 3.9 to be installed. At the time of writing, 3.9.19 is the latest version but you can determine this by running ```pyenv install -l```. Just put the correct version in the command below.
```
pyenv install 3.9.19
pyenv global 3.9.19
```

### Set up your virtual environment
```
cd /home/pi/UK-Bird-Classifier-Pt-2
python3 -m venv venv
```

### Return your system to default version of python
```
pyenv global system
```

### Install all required libraries
```
source venv/bin/activate
pip3 install -r <requirements.txt
```

### Copy in your source images
```
cp <image location / files> birds_model/train/
cp <images to validate model> birds_model/validate/
cp <new images to run the model against> test_data/
```

### Run labelImg and identify / label all birds in your images
```
labelImg
```
* Click 'Open Dir' and navigate to the birds_model/train directory
* Click 'Change Save Dir' and also set that to the birds_model/train directory
* Open the first image, click 'Create RectBox' and draw a box around the bird and provide a label for it
* Repeat for any other birds
* Click the 'Next Image' button and save your image. It'll then move on to the next one
* When you're done with all your training images, repeat by changing the source and save directories to birds_model/validate and labelling all images in there

### Build your data model
```
python build_model.py
```
(Make a cup of your favourite beverage - this will take some time)

### Run your data model against your test data
```
python test_model.py
```

### Check the results
Open up the images in the test_data_results directory and see how well your model has performed in identifying the birds!