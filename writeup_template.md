# **Behavioral Cloning**
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter and 5x5 filter sizes and depths between 24 and 64 (model.py lines 40-44)

The model includes RELU layers to introduce nonlinearity (for the first 3 convolutional layers), and the data is normalized in the model using a Keras lambda layer (code line 38).

The model also includes 1 cropping layer to crop the top 70 pixels and bottom 25 pixels from the original data.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 52). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 51).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to copy the architecture from some success models.

At the begining, I studied from LeNet, however, it is not enough for 160x320x3 pictures, we need more powerful network than it.
Then I tried AlexNet, which has the similiar size's input with our case. However, as we do not have so much training data, AlexNet will easily lead the model to
overfitting, which also do not give good result. Finally, I chose the NVIDIA's self driving car model, which gives the best result.

#### 2. Final Model Architecture

The final model architecture (model.py lines 37-49) consisted of a convolution neural network of 5 convolutional layers and 4 fully connected layers.
The first 3 convolutional layer are having the kernel size of 5x5 and strides of (2,2), and the number of kernels are 24, 36 and 48, they are all followed by relu layer. The following 2 conv layers
 are having 64 kernels with kernel size 3x3. The 4 fully connected layers's size are 100, 50, 10 and 1.



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.


To augment the data sat, I also flipped images and angles. Also I used the pictures from all the 3 cameras so that I would have more data.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by several trials I used an adam optimizer so that manually training the learning rate wasn't necessary.
