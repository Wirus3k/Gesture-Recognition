### Gesture-Recognition-Using-Image-Processing

The aim of the project is to create a window application that will enable finger counting 
of one hand in real time. The work will implement appropriate 
image-processing algorithms so that, as a result of the operation, the output will be 
hand shape will be obtained as the output. The next step will be to apply an algorithm that, from the perimeter
will be able to detect the number of fingers shown by the user. The entire programme will be 
made on the basis of a single OpenCV library for tasks related to 
image processing.

Image processing algorithm with description

# Approach 1 - main approach, using the following steps to process the ROI window:
1. Conversion of the image from BGR to greyscale format.
2. Gaussian blurring to remove noise.
3. calculation of the absolute difference between the current frame and the background.
4. image rectification.
5. outline drawing.





