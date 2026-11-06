import cv2
from recognition.face_engine import FaceEngine
from database.database import(
    initialise_database,
    get_all_embaddings,
    add_person
)

from registration.register import capture_embadding


def prompt_for_name(camera):
    """Read a name in the camera window so focus stays with OpenCV."""
    name = ""
    while True:
      ret, frame = camera.read()
      if not ret:
        return None

      cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 105), (0, 0, 0), -1)
      cv2.putText(frame, "Type a name, then press Enter (Esc cancels)",
                  (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
      cv2.putText(frame, name + "_", (20, 82), cv2.FONT_HERSHEY_SIMPLEX,
                  0.75, (0, 255, 255), 2)
      cv2.imshow("Camera Feed", frame)

      key = cv2.waitKey(1) & 0xFF
      if key in (10, 13):
        return name.strip()
      if key == 27:
        return None
      if key in (8, 127):
        name = name[:-1]
      elif 32 <= key < 127 and len(name) < 32:
        name += chr(key)


def main():
    initialise_database()
    engine=FaceEngine()
    saved_faces=get_all_embaddings()

    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
      print("Can't access this webcam")
      return
    
    print("Click the Camera Feed window before using its keys.")
    print("Press Q to quit or R to register a person.")
    
    while True:
      ret, frame = camera.read()
      if not ret:
        print("Can't capture the frame")
        break

      faces=engine.detect_faces(frame)
      for face in faces:
        x1,y1,x2,y2=map(int,face.bbox)
        embadding=engine.get_embaddings(face)
        match=None
        if embadding is not None and saved_faces:
          match=engine.recognise(embadding,saved_faces)
        if match:
          name=match["name"]
          color=(0,255,0)
          label=f"{name}({match['score']:.2f})"
        else:
          name="unknown"
          color=(0,165,255)
          label="Unknown person"

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
      cv2.putText(frame,"R:Register|Q:Quit",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

      cv2.imshow('Camera Feed', frame)

      key = cv2.waitKey(1) & 0xFF
      if key < 128:
        key = ord(chr(key).lower())
      if key == ord('q'):
        break

      elif key==ord('r'):
        if len(faces)!=1: 
         print("Keep exactly one face visible.")
         continue
        name = prompt_for_name(camera)

        if not name:
          print("Name entry cancelled or empty.")
          continue
        embaddings=capture_embadding(camera,engine,total_samples=15)
        if not embaddings:
          print("Registration cancelled")
          continue

        existing=engine.recognise(
          embaddings[0],
          saved_faces
        )

        if existing:
          print(f"Already registered as:{existing['name']}")
          continue

        person_id=add_person(name,embaddings)

        print(f"Registration successful! ID:{person_id}")

        saved_faces=get_all_embaddings()

        

    camera.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
  main()
