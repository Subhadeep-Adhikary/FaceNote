import cv2
import numpy as np
from recognition.face_engine import FaceEngine

engine = FaceEngine()

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    if not ret:
        break

    faces=engine.detect_faces(frame)

    for face in faces:
        x1,y1,x2,y2=map(int,face.bbox)

        embadding=engine.get_embaddings(face)

        if embadding is not None:

            print("Embadding shape: ",embadding.shape)

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),2
                )
            cv2.putText(frame,"Face Detected",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x') or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()