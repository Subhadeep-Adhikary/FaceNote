import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        print("Face recognisation model loaded")

    def detect_faces(self, frame):
        faces = self.app.get(frame)
        return faces

    def get_embaddings(self,face):
        embedding=np.asarray(
            face.embedding,
            dtype=np.float32
        )

        norm=np.linalg.norm(embedding)
        if norm==0:
            return None

        return embedding/norm

    def cosine_similarity(self,emb1,emb2):

        emb1=emb1/np.linalg.norm(emb1)
        emb2=emb2/np.linalg.norm(emb2)

        return float(np.dot(emb1,emb2))

    def recognise(self,new_embedding,saved_faces):
        best_match=None
        best_score=-1
        for saved in saved_faces:
            score=self.cosine_similarity(new_embedding,
                                         saved["embedding"])
            if score>best_score:
                best_score=score
                best_match=saved

        THRESHOLD=0.45

        if best_match and best_score>=THRESHOLD:

            return{
                "id":best_match["id"],
                "name":best_match["name"],
                "score":best_score
            }
        return None           
