import cv2

def capture_embadding(cap,engine,total_samples=15):
    embaddings=[]
    print("\nRegistration started!")
    print("Look at the camera.")
    print("Click the Face Registration window, then press S to capture each sample.")
    print("Press Q in that window to cancel.")

    while len(embaddings)<total_samples:
        ret,frame=cap.read()

        if not ret:
            break

        faces=engine.detect_faces(frame)

        for face in faces:
            x1,y1,x2,y2=map(int,face.bbox)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame,f"Sample {len(embaddings)}/{total_samples}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.imshow("Face Registration",frame)
        key = cv2.waitKey(1) & 0xFF
        if key < 128:
            key = ord(chr(key).lower())
        if key==ord('q'):
            return None
        if key==ord('s'):
            if len(faces)!=1:
                print("Please keep exactly one face visible.")
                continue
            face=faces[0]
            x1,y1,x2,y2=face.bbox
            if(x2-x1)<100 or (y2-y1)<100:
                print("Move closer to the camera.")
                continue
            embedding=engine.get_embaddings(face)
            if embedding is not None:
                embaddings.append(embedding)
                print(f"Captured {len(embaddings)}/{total_samples}")
    cv2.destroyWindow("Face Registration")
    
    return embaddings
