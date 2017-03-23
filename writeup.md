# Project#5 - Vehicle Detection 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
##Project Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

[x] Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
[x] Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
[x] To normalize your features and randomize a selection for training and testing.
[x] Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
[x] Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
[x] Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/mediumScale.PNG
[image4]: ./output_images/smallerScale.PNG
[image5]: ./output_images/withTest1.png
[image6]: ./output_images/withTest3.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Considered the rubric points individually and described how those are addressed in the implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Yes, You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The function `extract_features()` in the lesson_functions.py  has all the implementation.
First it converts the image to YUV color space. Then resizes the image to 16x16 size where the flattened pixels for one set of features. Even the histogram (with 32 bins) is collected for all channels and are used as another set of features. Then the HOG of all the channels, with 9 orientations, 16x16 pixels per cell and 2x2 cells per block, is computed and all the features are concatenated to from the final feature vector. 

This process is carried for both Vehicle and Non-Vehicle images to form the feature dataset.
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally settled on
```
spatial_size = (16,16)
hist_bins = 32
orientations = 9
pixels in cell = 16x16
cells in block = 2x2
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using using a train and test split of 80:20 percentage chosen randomly.
Where training the SVM classifier:
```
(carnd-term1) C:\Users\vkrishnam\Udacity\P5\CarNDVehicleDetection>python train_SVMClassifier.py
95.85 Seconds to extract HOG features...
Using: 9 orientations 16 pixels per cell and 2 cells per block
Feature vector length: 1836
6.66 Seconds to train SVC...
Test Accuracy of SVC =  0.9966
My SVC predicts:  [ 0.  0.  1.  1.  0.  1.  1.  1.  1.  1.]
For these 10 labels:  [ 0.  0.  1.  1.  0.  1.  1.  1.  1.  1.]
0.001 Seconds to predict 10 labels with SVC
Writing the SVM model and Scaler model to pickle file...
```

Once trained both the Classifier model and the Scaler model is written to a Pickle file to be used later on in the Detection Pipeline.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Two different sliding windows with different scales are choosen after lot of iteration. First scale searches for cars in medium vicinity and Second scale searches for cars little far:

Medium Scale Search Window scheme
![alt text][image3]


Small Scale Search Window scheme
![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
![alt text][image6]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As explained in the lessons, I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I also keep record of the prev frame detections and use them in the heatmap to remove the false positive and also to have more area and smooth bounding boxes drawn from frame to frame.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It seems HOG of Luma alone might not work very satisfactorily although we human feel intuitively we should be able to detect objects with luma and its edges. Also another challenge is to manage te tradeoff between the number of search points and scales whic might exponentially increase the compute time versus the accuracy of detection which can only be arrived after lot of trails. When the vehicles/objects are really small it becomes very challenging to detect, no matter after upscaling
the real needed HOG features seem to be lost for successful detections. Pipeline is likely to fail in the scenarios of small vehicles and occluded vehicles. The pipeline can be made more robust with more data used for training and also a more robust features and classifier or may be employ some deep learning techniques here too for detections. 

