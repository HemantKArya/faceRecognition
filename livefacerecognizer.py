import cv2
import os
import numpy as np 
import FaceRecognition as fr 


#capture image from web cam first
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')

name = {0: "unknown",1: "Hemant"}

cap = cv2.VideoCapture(0)   #capture image from web cam

while True:
    ret, test_img = cap.read()
    faces_detected,gray_img = fr.faceDetection(test_img)

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=6)

    resized_img =cv2.resize(test_img,(1000,700))
    cv2.imshow('FAce detection tutorial',resized_img)
    cv2.waitKey(10)

    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+w,x:x+h]
        label,confidence = face_recognizer.predict(roi_gray)
        print(confidence,label)
        fr.draw_rect(test_img,face)
        predicted_name = name[label]
        if confidence < 60:
            fr.put_text(test_img,predicted_name,x,y)

    resized_img = cv2.resize(test_img,(1000,700))
    cv2.imshow("FaceREcognition",resized_img)
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    



        