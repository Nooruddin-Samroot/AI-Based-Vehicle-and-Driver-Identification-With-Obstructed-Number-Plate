This a AI based system which detects vehicle, Number plate, and face using YOLOv11 model.
then it tracks vehicle using SORT
to smooth the tracking we have interpolation and buffering
After that we have used EasyOCR to extract text from the number plates cropped by yolo model.
To make easyOCR perfect we have Fuzzzy matching which try different matches with the database.
we have also used vehicle recognition using CLIP module and face recognition using DeepFace and InsightFace
we used Sqlite3 database for storing pre-existing data and real-time data as well.
All this is displayed in a User friendly interface using Streamlit.
