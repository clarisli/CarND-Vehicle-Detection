## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car.png
[image2]: ./examples/not_car.png
[image3]: ./examples/car_hog.png
[image4]: ./examples/not_car_hog.png 
[image5]: ./examples/spatial.png 
[image6]: ./examples/color_hist.png 
[image7]: ./examples/normalize.png
[image8]: ./examples/pipeline.png
[image9]: ./examples/multi_scale.png
[image10]: ./examples/slide_window.png
[image11]: ./examples/heatmap.png
[image12]: ./examples/test_video.gif

![alt text][image12]

### Dataset

I trained the model using the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples. These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

To train the model, download the labeled under `./train_data`.

---

### Data Exploration

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some examples of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

### Feature Extraction

#### 1. Histogram of Oriented Gradients (HOG)

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are the examples using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image4]

I further speed up this step using the HOG sub-sampling technique. I did this by extracting HOG features at once in a small set of predetermined window sizes(defined by a scale argument), then do sub-sampling to get all of its overlaying windows. The code for sub-sampling is in `find_cars()` in cell 27 of `vehicle_detection.ipynb`.

#### 2. Spatial Binning

In addition, I used spatial binning as another feature. I scaled down the 64x64 image to 8x8 and was able to still preserve its relevent features. 

Here's a comparision between a car and a non-car's spatial binning features:

![alt text][image5]

I did this using OpenCV's `cv2.resize()` in cell 11 of `vehicle_detection.ipynb`.


#### 3. Color Histogram

I also included the color histogram feature with 8 histogram bins and a range between 0.1 and 0.5 in YCbCr color space.

Here's a comparision between a car and a non-car's color histogram features:

![alt text][image6]

I did this in cell 13 of `vehicle_detection.ipynb`.

#### 4. Feature parameters

The goal was to tune the parameters to maxmize the accuracy while minimize the fitting/feature extraction time. I tried various combinations and settled to following:

| Parameter     | Value         | 
| ------------- |:-------------:|
| color space   | YCrCb         |
| orientations  | 9             |
| pix_per_cell  | 8             |
| cell_per_block| 1             |
| hog_channel   | ALL           |
| spatial_size  | (8,8)         |
| hist_bins     | 8             |
| hist_range    | (0.1,0.5)     |

I obtained a training accuracy of 0.9963 and test accuracy of 0.9955 with these parameters. It took 0.00084s to predict 10 labels.

I implemented this step in cell # of `vehicle_detction.ipynb`.

#### 5. Train the classifier

I used the `RobustScaler()` from `sklearn` package to normalize the data, then trained a linear SVM using a combination of HOG, spatial binning, and color histogram features. 

Following is an example the raw and normalized features:

![alt text][image7]

I did this in cell 16 of `vehicle_detction.ipynb`.


### Sliding Window Search

#### 1. Implementation

The algorithm steps across an image in a grid pattern and cut out a window at each step. Then it extracts features on each window, uses the classifier trained above to give a prediction at each step - identifying which windows contain a car, and which windows do not contain a car. To speed up the algorithm, I restricted the search only to the areas where vehicles might appear. 

Here are the output of my first attempt of sliding window search:

![alt text][image10]

I did this in `slide_window()` in cell 23 of `vehicle_detection.ipynb`. 

#### 2. Multi-scale windows

To speed up the algorithm, I searched under multiple scales. I first defined a minimum and maximum scale at which I expect the vehicle to appear in the image, then came up with few reasonable intermediate ranges to scan as well. For example, vehicles near horizon are generally small, so I searched in a small scale near that region with a narrow strip.

Here's an example of the multi-scale windows:

![alt text][image9]

I did this in cell 33 of `vehicle detection.ipynb`.


#### 2. Pipeline Results

Ultimately I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image8]


---

### Video Implementation

#### 1. Final video ouput
Here's a [link to my video result](./test_videos_output/project_video.mp4)


#### 2. Multiple detections and false positives
When the classifier reports postive detections, there are very likely to be multiple of them overlapping with each other on the vehicle. False positive detections are found on the guardrail to the left, yellow lane line, and the shadows on road surface. 

By adding "heat"(+=1) for all positive detections, I built heap-maps to overcome these two problems. The "hot" parts of the map are where the cars are, and by imposing a threshold, I can reject areas affected by false positive. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I did this in cell # of `vehicle_detection.ipynb`.

Here are six frames and their corresponding heatmaps, output of `scipy.ndimage.measurements.label()` on heatmaps, and the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image11]


---

### Discussion

#### 1. Problems / issues

I faced three main problems in my implementation: slow search, false postives, and multiple detections. 

To overcome, I optimized the sliding window approach with HOG sub-sampling, multi-scale windows, and limited the search area to the region where the cars are most likely to occur, also tuned the feature parameters to reduce number of features while maintaining a reasonable accuracy. 


#### 2. Future works

The system might fail in real time, it took around 8 minutes to finish processing a 1 minute video. 

I might want to try the following in future:

* add vehicle detection to the Advanced Finding Lane Line Project
* use deep learning approach for vehicle detection
* augment the training data with [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations)
