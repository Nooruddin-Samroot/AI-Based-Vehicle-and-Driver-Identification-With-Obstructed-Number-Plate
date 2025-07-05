import json
from deepface import DeepFace

# Verify it two images are of the same person
result = DeepFace.verify("Picture 1.jpg", "Test 1.jpeg") 

# Find face in database
dfs= DeepFace.find(img_path = "Picture 2.jpg", db_path="C:/Users/Kriti/.ms-ad/face_recognition")
print(dfs)

# Face analysis (age, gender, emotion, race)
objs = DeepFace.analyze("Picture 4.jpeg")
print(json.dumps(objs, indent=4))









