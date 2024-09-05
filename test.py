import cv2
import numpy as np
from picamera2 import Picamera2

def nothing(x):
    pass

# Initialize Picamera2 object
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Set HSV range (adjust to the desired color range)
lower_range = np.array([]) # fill in RED value [L-H, L-S, L-V] 
upper_range = np.array([])  # fill in RED value [U-H, U-S, U-V] 
lower_range1 = np.array([]) # fill in BLUE value [L-H, L-S, L-V] 
upper_range1 = np.array([])  # fill in BLUE value [U-H, U-S, U-V] 

def red(img):
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert from BGR to HSV color space    
    mask = cv2.inRange(hsv, lower_range, upper_range) # Create a mask using the HSV range
    _, mask1 = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) # Binarization
    cnts, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Contour detection

    # Draw rectangles around contours that exceed a specific area
    for c in cnts:
        if cv2.contourArea(c) > 600:  # Process only objects that exceed a certain area
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "DETECT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def blue(img):
     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert from BGR to HSV color space    
    mask = cv2.inRange(hsv, lower_range1, upper_range1) # Create a mask using the HSV range
    _, mask1 = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) # Binarization
    cnts, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Contour detection

    # Draw rectangles around contours that exceed a specific area
    for c in cnts:
        if cv2.contourArea(c) > 600:  # Process only objects that exceed a certain area
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "DETECT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            
while True:

    frame = picam2.capture_array()   # Capture frame with Picamera2   
    red(frame)
    blue(frame)
    # in case Red and Blue appears inverted in the display
    # frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # Display the frame
    cv2.imshow("FRAME", frame)

    # Exit when 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cv2.destroyAllWindows()
picam2.close()
