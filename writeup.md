# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/ycrcbhog.png
[image2]: ./output_images/windo.png

---

### Files
The main project file is `main.py`, and the methods used are in `util_functions.py`.

### Feature extraction
In order to train a classifier to detect car vs non-car images, 8792 examples of 64x64 car images and 8968 examples of 64x64 non-car images were used for training. 

The feature extraction pipeline for all images was as follows:

* first, the image was converted to YCrCb color space; I also tried LUV, but the results were similar
* HOG features were extracted for all three channels of the YCrCb image, with: 9 orientations, 8 pixels per cell, and 2 cells per block for the normalization step in the HOG transform.
* Spatial features were also obtained after rescalling the image to 16x16 format and ravelling the image to a one-dimensional vector
* The histograms of the three channels of the image were also obtained, with 16 bins
* All the features were concatenated in a one-dimensional vector for each image.

An example of the original image, converted to YCrCb and the resulting HOG transforms of the first two channels is shown below. 

![alt text][image1]

### Classifier

The feature matrix of all the images was scaled such that each future vecture has zero mean and unit variance, using the scikit StandardScaler() method.

The scaled features were used to train linear support vector classifier provided by the scikit library. The labels were set to 1 for car images and to 0 for non-car images.

Using a 80%-20% data split for training-test data, the classifier achieved consistent test accuracty of between 97.5 - 98.8 percent for multiple runs. 

### Sliding windows for prediction

In the `find_cars` method, a sliding window approach was used to analyze different regions of the input image in order to check if a car is present or not. Windows of scales between 0.8 and 3 (with 10 levels in-between) were used. A window of scale of 1 is given by a 8x8 pixels cell. For scales of size smaller than 1.2, the region of interest is restricted to the further end of the road, where cars are likely to appear smaller due to the distance. For all windows, the lower half of the image is used.

In the X direction, the window step is one cell, whereas for the Y direction the window step is 2 cells, in order to speed up the processing time. Also, cars are moving more in the X direction than Y in the video file. 

The region inside each window is resized to 64x64 pixels, the features are extracted and scaled, and the result is used to make a prediction. If the prediction is 1 (car is present), the region inside the window is added to a heat map. Once windows of all the required scales are used, the heatmap values above a certain value are kept (threshold value is 7 now). This method reduces the number of false positives in the predictions. In order to identify the number of cars, the scikit `label` method identifies the number of connected components in the thresholded heatmap. Finally, for each connected component, the largest rectangle covering the region is used as the bounding box around a car in the image. 

Below, an example of different windows that have a positive prediction for the car (top left), the resulting heatmap (top right), the thresholded heatmap (bottom left), and the final bounding box for the car (lower right) are shown.

![alt text][image2]

### Output video

The final processed video is found in the `project_video_output.mp4` file.

### Discussion

The threshold value for the heatmap proved to be very important parameter to tune. A higher value will result in fewer false positives, but with a much tighter bounding box. 

A possible improvement to reduce the false positives even further is to look at a window of video frames, and check if a detected car appears again in the next frames, within a certain region around the detected car in the current frame. This would also result in a smoother bounding box, through averaging between frames. However, this technique would result in a delay in detecting a car, which might be a problem in case of emergencies.

 There is a small positive on the ground for a split second, at around second 40, due to some shadows that bring sturcture to the road. This could have been eliminated using the techique mentioned in this paragraph.

In order to speed up the processing time, I would first start with a large scale window, and if a car is detected, refine the search with smaller and smaller windows, in order to make sure the detection is not a false positive, and also to identify the bounding box for the car. 

