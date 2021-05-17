import cv2

#Select the default data source (Webcam) to acquire
cap = cv2.VideoCapture(0)

while True:
    #Get the image of each frame
    ret, frame = cap.read()

    #Apply some effect here

    #Display image
    #I want to display to a virtual webcam
    if ret:
        cv2.imshow("Window Name", frame)

    #Display 30 frames and end the display when the Enter key is pressed.
    if cv2.waitKey(30) >= 0:
        break

#End processing
cv2.destroyAllWindows()
cap.release()