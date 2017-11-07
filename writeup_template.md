## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_calibration.png "Undistorted"
[image2]: ./test_images/test5.jpg "Road Transformed"
[image7]: ./output_images/undistorted_image.png "Undistorted"
[image3]: ./output_images/threshold_image.png "Binary Example"
[image4]: ./output_images/perspective_transform_image.png "Warp Example"
[image5]: ./output_images/lane_lines.png "Fit Visual"
[image6]: ./output_images/final.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "advance_lane_lines.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I used the `undistort_image()` contained in the 2nd code cell of the notebook to undistort the following image 
![alt text][image2]

here is the result
![alt text][image7]



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.The `transformations()` in the 3rd code cell is used to create a binary image.Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in the 4th code cell of the IPython notebook.  The `perspective_transform()` function takes as inputs an image (`image`).  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[250,690], [580,460], [710,460], [1070,690]])
dst = np.float32([[250,720], [250,0], [1030,0],[1030,720]])

```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 250, 690      | 250, 720      |
| 580, 460      | 250, 0        |
| 710, 460      | 1030, 0       |
| 1070, 690     | 1030, 720     |


![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions `lane_boundary()` and `polyfit_using_prev_fit()`, which identify lane lines and fit a second order polynomial to both right and left lane lines, are in 5th and 6th code cell of the Jupyter notebook. The first of these computes a histogram of the bottom half of the image and finds the bottom-most x position (or "base") of the left and right lane lines. Originally these locations were identified from the local maxima of the left and right halves of the histogram. The function then identifies 9 windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image. Pixels belonging to each lane line are identified and the Numpy `polyfit()` method fits a second order polynomial to each set of pixels.


![alt text][image5]

The `polyfit_using_prev_fit()` function performs basically the same task, but alleviates much difficulty of the search process by leveraging a previous fit (from a previous video frame, for example) and only searching for lane pixels within a certain range of that fit. Pixels belonging to each lane line are identified and the Numpy `polyfit()` method fits a second order polynomial to each set of pixels.



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function `draw_lane_boundaries()` in the 7th code cell.
```
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
```
and position of vehicle is calculated using 
```
    middle = (left_fitx[-1] + right_fitx[-1])//2
    veh_pos = image.shape[1]//2
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    dx = (veh_pos - middle)*xm_per_pix # Positive on right, Negative on left

```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `draw_lane_boundaries()` in the 7th code cell.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems I faced were mostly due to diffrent lightning conditions, different lane colors and marks on the roads and shadows on the lanes and road. 

The pipeline is having trouble where lane color is not very different from the road. Also the perspective transform needs to be made robust to capture better viewing area specially in case of sharp turns.
