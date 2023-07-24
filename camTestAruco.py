import cv2
from timeit import default_timer as timer
import object_detector as ob
import numpy as np
import time

sum=0
count=0
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
# define a video capture object
vid = cv2.VideoCapture(0)
#vid = cv2.VideoCapture('shapesTestFinal.mp4')
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
   
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
#result = cv2.VideoWriter('shapesTestOp.mp4', 
#                         cv2.VideoWriter_fourcc(*'MP4V'),
#                         60, size)
while(True):
    start = timer()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #time.sleep(0.1)
    # Capture the video frame
    # by frame
    ret, img = vid.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img_edges = cv2.Canny(img,100,200)
    #img_edges = ResizeWithAspectRatio(img_edges,height=1000)
    #cv2.imshow('',img_edges)
    #cv2.waitKey(0)
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Load the image
   


    # Detect ArUco markers
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    # Draw detected markers on the image

    if markerIds is not None:
        #cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)

        # Iterate over detected markers
        for i, corners in enumerate(markerCorners):
            # Access the corners of the first marker
            corners = corners[0]

            # Calculate the perimeter
            marker_perimeter = cv2.arcLength(corners, True)

            # Access the marker ID
            marker_id = markerIds[i][0]
            #for j in corners:
            #    x,y = j.ravel()
            #    cv2.circle(img,(x,y),10,(0,0,255),-1)

            # Perform further processing for each marker
            # ...

    else:
        print("No markers detected in the image.")
        img_re = ResizeWithAspectRatio(img,height=700)
        cv2.imshow('frame', img_re)
        #result.write(img)
        continue


    # Draw polygon around the marker
    #int_corners = np.int0(markerCorners)
    #cv2.polylines(img, int_corners, True, (0, 255, 0), 5)


    # Aruco Perimeter
    aruco_perimeter = cv2.arcLength(markerCorners[0], True)
    #print(aruco_perimeter)

    image_edges = cv2.Canny(img, 0,255)




    # Pixel to mm ratio
    pixel_mm_ratio = aruco_perimeter / 100.9

    # Draw contours
    img_contours= img
    detector = ob.HomogeneousBgDetector()
    contours = detector.detect_objects(img)
    #print(contours)
    centerpoints = {}

    # Draw objects boundaries
    for  i,cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, True)
        perimeter = round(perimeter, 4)
        obj_peri= perimeter/pixel_mm_ratio
        #print(obj_peri)
        M = cv2.moments(cnt)
        if M['m00'] != 0.0:
            x1 = int(M['m10']/M['m00'])
            y1 = int(M['m01']/M['m00'])
            area = cv2.contourArea(cnt)
        
        
        # Get rect
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect

        # Get Width and Height of the Objects by applying the Ratio pixel to cm
        object_width = w / pixel_mm_ratio
        object_height = h / pixel_mm_ratio
        
        # Display rectangle
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rect_peri= 2*(w/pixel_mm_ratio) + 2*(h/pixel_mm_ratio)

        coordinate=list(centerpoints.values())
        #cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        #cv2.polylines(img, [box], True, (0, 0, 255), 2)
        #cv2.putText(img, "Width {} mm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        #cv2.putText(img, "Height {} mm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        #cv2.putText(img, "Width {} pixels".format(round(w, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        #cv2.putText(img, "Height {} pixels".format(round(h, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        cv2.putText(img, "Perimeter {} mm".format(round(obj_peri, 1)), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        #cv2.putText(img, "Perimeter {} mm".format(round(rect_peri, 1)), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        img = cv2.drawContours(img_contours,contours,-1,(255,0,0),2)
        


    
    
    # Display the resulting frame
    img_re = ResizeWithAspectRatio(img,height=700)
    cv2.imshow('frame', img_re)
    #result.write(img)
    end = timer()
    #result=(end -start)
    #print(result)
    sum= sum+(end-start)
    count=count+1
    
        
    
# After the loop release the cap object
print(sum/count)
vid.release()
#result.release()
# Destroy all the windows
cv2.destroyAllWindows()