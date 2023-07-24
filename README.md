# Object_dimension_detection

This software detects the perimeter of objects in millimeter in an image. This is achieved using an Aruco marker in the image which is used to calculate the perimeter of other objects in the image

### Implementation
The OpenCv library is used to find the contours of objects in the image. The perimeter of the contour of the Aruco marker(in pixels) in the image is compared to the actual perimter(in mm) of the aruco marker. This helps establish the pixels/mm ratio in the image, which is used to find the perimeter of other objects with their contours.

Camera Calibration is not used in this software as the images are taken perpendicularly from the top of the objects. However , the distortion matrix is obtained by running the `camCalib.py` file which requires a checkerboard pattern for calibration. The results are stored in a `.yaml` file, which can be used to modify the code as needed

An implementation provided in the `obj_detect_vid.py` which can be used to detect object dimensions in a video or live stream from a camera. The video is processed frame by frame. This software can process upto 25 frames per second , fps can be improved with better hardware acceleration or higher graphical computation mechanisms.
