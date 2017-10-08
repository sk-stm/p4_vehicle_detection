## Writeup for: P4 Vehicle Detection

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
[image_hog0]: ./output_images/hog_ch_0.png
[image_hog1]: ./output_images/hog_ch_1.png
[image_hog2]: ./output_images/hog_ch_2.png
[image_pipeline1]: ./output_images/raw_input.png
[image_pipeline2]: ./output_images/raw_detections.png
[image_pipeline3]: ./output_images/heatmap_raw.png
[image_pipeline4]: ./output_images/heatmap_merged.png
[image_pipeline5]: ./output_images/heatmap_merged_thresholded.png
[image_pipeline6]: ./output_images/filtered_detections.png
[image5]: ./output_images/.png
[image6]: ./output_images/.png
[image7]: ./output_images/.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `get_hog_features` function in the lesson_functions.py.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle`:

![alt text][image0]

and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(3, 3)`, showing a sub-region of a frame from the project video:

**Y-Channel**

![image_hog0]

**Cr-Channel**

![image_hog1]

**Cb-Channel**

![image_hog2]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and startet with low values (`orientation=4`, `pixels_per_Cell=4`, `cells_per_block=2`). I then gradually increased the values separately and trained the classifier, while keeping the classifier parameters fixed. I experienced an increase in accuracy. I increased the values until the accuracy started to decrease again. Finally, I choose the `L1-sqrt` block norm, as it resulted in a better test accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained n using the full data set you provided ([vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)). I used the scikit learn `fit` function to fit my parameterised algorithm to the training data. For prediction i use the `predict` function of that classifier. This is done in the `train_classifier` function in my train.py.

I also saved the trained classifier to a pickle file so I don't have to train it all the time when playing with the detection-part of the pipeline. This is done at the end of the `main` function in train.py.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For the sliding windows I used a customized version of the  `find_cars`-function that you suggested in the lesson. The scales for the windows were chosen empirically and looking at the window sizes in the test images. I thereby chose the scales [1,1.5,2,3,4] since they seemed to cover all the scales of cars in the video. I chose the overlap to be 75% which also was obtained empirically.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline first loads a frame from the video and then slides windows with different scale over the defined areas of the image. Here are the scales with the corresponding y-min and y-max values (read: (scale, y-min, y-max)):
[(1.0, 400, 528), (2.0, 384, 576), (3.0, 360, 700), (4.0, 380, 720)]. In x-dimension the whole image was searched all the time.

For each sampled window, the fetures were computed identical to the ones while training. The resultig feature vector was then classified with the SVM classifier and if the prediction was positive, the corresponding window-rectangle was added to a heatmap image. This image was then thesholded (parameter determined empirically). The resulting image is the final heat map.

For smoothing the the detections, I merged the heatmap of the current detections with the (thresholded) heatmap of the previous frame with a certain ratio (parameter determined empirically). This reduced the number of single-frame false positives but also stabilized correct detections over time.

Here are some example images from different steps of the pipeline:

**input image**

![image_pipeline1]

**raw detections**

![image_pipeline2]

**raw heatmap**

![image_pipeline3]

**merged heatmap**

![image_pipeline4]

**thresholded heatmap**

![image_pipeline5]

**final detections**

![image_pipeline6]

To optimize the performance of my classifier I searched for optimal paramters of the SVM using grid search for linear and rbf kernel SVMs. It was no surprise, that the the rbf kernel resulted in better accuracy. During my first gridsearch I also searched for the `C` paramter in [1,..,10]. The best value was `1` so I extended the search to [0.1,...,1] and the best values was still `1`.

However the `best` performing SVM on the training and test data was not the best performing SVM for the project video. I experienced a big gain in presicion (no false positives through out the entire image) but a big drop in recall (only 1 or two subwindows detected a car per frame, even when more than one car was clearly visible.)

Since the prediction of the rbf kernel in the project video contained very few predetions of bounding boxes (usually 1 or 2 in each frame) I fiddled with the parameters to regularise the SVM more. The problem was, that the actual performance of the SVM on the training and test set was not the same as the performance on the test images from the video. That means, that the accuracy gained during training was not helping a lot in determining if the classifier will actually perform good on the video. So I couldn't use a grid search algorithm or similar to evaluate the performance of the classifier, but had to judge the outcome video by my self every time, and tune the parameters by hand. So development speed was very slow. I finally found it easier to use the Linear SVM, that had poorer precision but a better recall, an then deal with the false positives with smoothing and thresholding.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_imagees/project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. `scipy.ndimage.measurements.label()` was used to identify individual blobs in the heatmap. I assumed that each blob corresponded to a vehicle. Finally I constructed bounding boxes to cover the area of each blob detected. Also the heatmap was merged with the one from the previous frame, as described above. You'll find the filtering in the `non_maximum_suppression` method.


### Here is one frames and it's corresponding heatmap:

frame:
![alt text][image_pipeline1]

the thesholded heatmap looks like this:

![alt text][image_pipeline5]

### Here the resulting bounding boxes:
![alt text][image_pipeline6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A big issue durign this project was the parameter search for a "well performing" SVM. Since the data set was very indiverse, an SVM with a high test accuracy was not necessarily predicting cars in the test- and project video with a satisfying precision/recal. It was easier to use an SVM with a higher recall value and then use heat map smoothing accross the frames instead of optimising the paramters of the rbf-kernel SVM. 

That also means that the prediction will definetely become better if the training set would become more diverse and bigger since more examples will benefit the prediction. Also other prediction algorithms like `Decision Trees` could be used. But in the end everything comes down to parameter tuning to get better and better performance with this pipeline.

Finally the algorithm doesn't work very well on images, that are different from what the classifier has seen from in the training data. Also I was very suppried how much "visual accuracy" one can gain with a good smoothing algorithm.

Currently each new appearing car in the scene will get another id. One could also increase the id assignment if each "blob" was also traced and tracked with a bayes filter (e.g. Kalman) with suitable motion-model.

As seen while the black car overtakes the white one, detections will merge if two vehicle overlap. Occlusion is next to impossible to overcome with the current pipeline. To solve this, the classifier could be trained on car-parts only, along with an offset-vector to the car's center. This would result in a General Hough pipeline. Alternatively, (r)RCNNs could help ..

Finally, Studient Gradient Descent is always a pain. Instead of selecting features by hand, one could should compute all of the common features and then use a Random Forest in order to select the most useful ones
