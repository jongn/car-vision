# car-vision
## What's this?
Computer vision is an important aspect in many industries, and I decided it was about time to familiarize myself with some basic concepts and practices. Luckily, image and video data exists everywhere on the internet, and I decided to begin somewhere basic: traffic cams. My initial goals are to (in order of triviality and priority):

1. At any time, give a good estimate of how heavy the traffic is
2. Be able to alert a user of unique events (accidents, emergency vehicles)
3. Identify areas of high "scrutiny" (how often speeders get pulled over)

These goals will be updated over time. Note that I will be using OpenCV (for python) along with some extra modules found in the [contribution repo](https://github.com/opencv/opencv_contrib).

## Classification
A commonly used strategy, for good reason. Vehicle detection is usually achieved using a Haar-like feature detector, which is both fast and efficient in recognizing commonly seen shapes. Read more about Haar-like features [here](https://en.wikipedia.org/wiki/Haar-like_features). Fortunately, OpenCV comes with [cascading classifiers](http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html), which we will take full advantage of.

### Training the classifier
The human element of machine learning: cascade classifiers learn by taking "positive" and "negative" images. To see exactly how it is trained (and do it yourself), see [here](https://github.com/mrnugget/opencv-haar-classifier-training). For now, I will opt out of this manual labor be using the already trained classifier found [here](https://github.com/andrewssobral/vehicle_detection_haarcascades).

## Background Subtraction
To avoid manually teaching the classifier, a different approach can be viable. In background subtraction, a foreground mask is generated in order to differentiate between static and dynamic objects in a frame. Essentially, as the name subjects, the background is "subtracted," leaving the dynamic objects (in this case, the cars). 


OpenCV (as of this date) provides [BackgroundSubtractorMOG2](http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html?highlight=backgroundsubtractorMOG2#backgroundsubtractormog2) by default. The two other subtractors what will be tested are [BackgroundSubtractorMOG](http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html?highlight=backgroundsubtractorMOG#backgroundsubtractormog) and [BackgroundSubtractorGMG](http://docs.opencv.org/ref/2.4/d8/d43/classcv_1_1BackgroundSubtractorGMG.html), found in the bgsegm module from the contribution repo. To learn more about the algorithms being used, see the additional resources section.


Since the background subtractors need to first learn the background, the first few frames will have a very low accuracy. This leads us to the first optimization.

### Optimization: Approximate the background beforehand
The current implementation will first approximate the background by using a running average of recent frames, and then apply that approximation to the background subtractor with some initial learning rate. To learn more about how this works, see the additional resources section.

## Some Images

## Additional resources
BackgroundSubtractorMOG algorithm source [paper](http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf)

BackgroundSubtractorMOG2 algorithm source [paper](http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf)

BackgroundSubtractorGMG algorithm source [paper](http://goldberg.berkeley.edu/pubs/acc-2012-visual-tracking-final.pdf)

Running average background subtraction algorithm source [paper](https://pdfs.semanticscholar.org/db2e/6623c8c0f42e29baf066f4499015c8397dae.pdf)
