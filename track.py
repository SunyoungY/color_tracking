import numpy as np
import cv2
from picamera2 import Picamera2

def nothing(x):
    pass

# Picamera2 setting
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# create window for trackbar
cv2.namedWindow("Trackbars")

# create trackbar
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    # capture frame
    frame = picam2.capture_array()

    # convert frame into HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # bring HSV value from trackbar
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    # create mask based on HSV range
    lower_hsv = np.array([l_h, l_s, l_v])
    upper_hsv = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # create result image using mask
    result = cv2.bitwise_and(frame, frame, mask=mask)
  
    # in case Red and Blue appears inverted in the display
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
  
    # print result
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)  

    # press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close window
cv2.destroyAllWindows()
picam2.close()
