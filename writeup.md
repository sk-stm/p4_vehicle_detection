##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image0]: ./output_images/car.png
[image1]: ./output_images/not_car.png
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `get_hog_features` function in the lesson_functions.py.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle`:

![alt text][image0]

and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSL` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(3, 3)`:

TODO:
![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and startet with low values (orientation=4, pixels_per_Cell=4, cells_per_block=2). I then gradually increased the values separately and trained the classifier, while keeping the classifier parameters fixed. I experienced an increase in accuracy. I increased the values until the accuracy started to decrease again.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the full data set of `cars` and `not cars` from https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip and https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip respectively. I used the scikit learn `fit` function to fit my parameterised algorithm to the training data. For prediction i use the `predict` function of that classifier.

This is done in the `train_classifier` function in my train.py.

I also saved the trained classifier to a pickle file so I don't have to train it all the time I want to detect and classify cars in an image. This is done at the end of the `main` function in my train.py.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For the sliding windows I used the in the lessons provided function `find_cars`. The scales for the windows I chose by experiment and looking at the window sizes in the test images. I thereby chose the scales [1,2,3,4] since they seemed to cover all the scales of cars in the video. I didn't changed the overlap of the windows from the implementation in the lessons because it seemed not necessary.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HLS 3-channel HOG features with `L1-sqrt` block norm, plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

TODO
![alt text][image4]

To optimize the performance of my classifier I searched for optimal paramters for the SVM using grid search for linear and rbf kernel SVMs. It was no surprise, that the better SVM classifier had the rbf kernel. During my first gridsearch I also searched for the `C` paramter in [1,..,10]. The best value was `1` so I extended the search to [0.1,...,1] and the best values was still `1`.

However the `best` performing SVM on the training and test data was not the best performing SVM for the project video. I experienced a big gain in presicion (no false positives through out the entire image) but a big drop in recall (only 1 or two subwindows detected a car per frame, even when more than one car was clearly visible.)

Since the prediction of the rbf kernel in the project video contained very few predetions of bounding boxes (usually 1 or 2 in each frame) I fiddled with the parameters to regularise the SVM more. The problem was, that the actual performance of the SVM on the training and test set was not the same as the performance on the test images from the video. That means, that the accuracy gained during training was not helping a lot in determining if the classifier will actually perform good on the video. So I couldn't use a grid search algorithm or similar to evaluate the performance of the classifier, but had to judge the outcome video by my self every time. and tune the parameters by hand. So development speed was very slow. I finally found it easier to use the Linear SVM, that had poorer precision but a better recall, an then deal with the false positives with smoothing.
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

TODO
![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

TODO
![alt text][image6]

### Here the resulting bounding boxes:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A big issue durign this project was the parameter serach for a "well performing" SVM. Since the data set was very indiverse, a SVM with a high test accuracy was not necessarily good enough to sufficiently predict cars in the test- and project video. It was easier to use a linear SVM with a higher recall value and then use heat map smoothing accross the frames instead of optimising the paramters of the rbf-kernel SVM. 

That also means that the prediction will definetely become better if the training set would become mor diverse and bigger since more examples will benefit the prediction. Also other prediction algorithms like `Decision Trees` could be used. But in the end everything comes down to parameter tuning to get better and better performance with this pipeline.

Finally the algorithm doesn't work very well on images, that are different from what the classifier has seen from in the training data. Also I was very suppried how much "visual accuracy" one can gain with a good smoothing algorithm.

Currently each new appearing car in the scene will get another id. One could also increase the id assignment if each "blob" was also traced with a kalman filter to detect the same car again after it was covered by another car.
