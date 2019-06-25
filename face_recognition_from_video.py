# Face Recognition

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') # We load the cascade for smiles eyes.

fileName='video.mp4' 

def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
       
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22) # We apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
        
        smiles = smile_cascade.detectMultiScale(gray, 1.7, 22) # We apply the detectMultiScale method to locate one or several smiles in the image.
        for (dx, dy, dw, dh) in smiles: # For each detected eye:
            cv2.rectangle(roi_color,(dx, dy),(dx+dw, dy+dh), (0, 104, 255), 2) # We paint a rectangle around the smiles, but inside the referential of the face.
    return frame # We return the image with the detector rectangles.

cap = cv2.VideoCapture(fileName) # We turn the webcam on.

    
while(cap.isOpened()):                    # play the video by reading frame by frame
    ret, frame = cap.read()
    if ret==True:
        # optional: do some image processing here 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
        canvas = detect(gray, frame) # We get the output of our detect function.
        cv2.imshow('Video Detection Video', canvas) 
    
        #cv2.imshow('frame',frame)              # show the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()    
    

