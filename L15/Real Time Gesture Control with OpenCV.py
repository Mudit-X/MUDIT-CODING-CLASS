import cv2
import numpy

#Set up webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    #Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    #Convert to HSV for color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Define the range for skin color in HSV
    lower_skin = numpy.array([0, 20, 70], dtype=numpy.uint8)
    upper_skin = numpy.array([20, 255, 255], dtype=numpy.uint8)
    
    #Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    #Apply the mask to the original frame
    skin = cv2.bitwise_and(frame, frame, mask=mask)

    #Find contours (hand shape) in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #If contours are found, draw them
    if contours:
        max_contour = max(contours, key=cv2.contourArea) #Get the llargest countour
        (x, y, w, h) = cv2.boundingRect(max_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #Green rectangle with thickness 2

        #Get the center of the hand for further tracking or interaction
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1) #Red circle with radius 5

        #Display the original and result frames
        cv2.imshow('Original', frame)
        cv2.imshow('Result', skin)

        #Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#Release the capture and close windows
cap.release()
cv2.destroyAllWindows()