import cv2


class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Gaussian Blur
        #blur = cv2.GaussianBlur(frame ,(9, 9), 0)
  
        # Median Blur
        #blur = cv2.medianBlur(frame, 3)
  
        # Bilateral Blur
        blur = cv2.bilateralFilter(frame, 7, 75, 75)

        # Convert frame to grayscale
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # Create a Mask with adaptive threshold
        thresh = 110
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
        #_,mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50 and area <3900:
                
                cnt = cv2.approxPolyDP(cnt, 0.008*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours

    # def get_objects_rect(self):
    #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    #     box = np.int0(box)