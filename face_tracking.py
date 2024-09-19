import cv2 as cv

# Load the Haar Cascade Classifier
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Open webcam feed
capture = cv.VideoCapture(0)


while True:
    # Capture frame-by-frame from the webcam
    isTrue, frame = capture.read()
    
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=-1,)
    
    # Display the resulting frame with face rectangles
    cv.imshow('Webcam Face Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
capture.release()
cv.destroyAllWindows()
