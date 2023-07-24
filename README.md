# Object_dimension_detection

This software detects the perimeter of objects in millimeter in an image. This is achieved using an Aruco marker in the image which is used to calculate the perimeter of other objects in the image

### Implementation
The OpenCv library is used to find the contours of objects in the image. The perimeter of the contour of the Aruco marker(in pixels) in the image is compared to the actual perimter(in mm) of the aruco marker. This helps establish the pixels/mm ratio in the image, which is used to find the perimeter of other objects with their contours.
