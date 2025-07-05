import cv2
import numpy as np
import os
import csv
from collections import defaultdict, deque
from ultralytics import YOLO
from sort import Sort
from scipy.interpolate import interp1d
import streamlit as st
import time
from TRACKING_OCR import track_objects
import threading
import queue
import easyocr
import sqlite3
from fuzzywuzzy import fuzz, process
from io import BytesIO
from PIL import Image
from vehicle_matching import (
    image_similarity_results,
    image_similarity_lock,
    start_image_similarity_thread,
    queue_similarity_task
)
import datetime  

video_path         = r"D:\CAPSTONE PROJECT\VIDEO\2103099-uhd_3840_2160_30fps.mp4"
vehicle_model_path = r"D:\CAPSTONE PROJECT\MODELS\VEHICLE_TYPE_MODEL\best.pt"
plate_model_path   = r"D:\CAPSTONE PROJECT\MODELS\NUMBER_PLATE_MODEL\best.pt"
face_model_path    = r"D:\CAPSTONE PROJECT\MODELS\FACE_MODEL\best.pt"
csv_path           = os.path.join(os.path.dirname(video_path), "live_interpolated_bboxes.csv")

RESIZE_WIDTH  = 1280
RESIZE_HEIGHT = 660
BUFFER_SIZE = 100
INTERPOLATION_DELAY = 5

CONF_THRESH = {"vehicle": 0.7, "plate": 0.4, "face": 0.3}

vehicle_model = YOLO(vehicle_model_path)
plate_model   = YOLO(plate_model_path)
face_model    = YOLO(face_model_path)

vehicle_class_names = vehicle_model.names  
vehicle_class_map = {}  

tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)

track_frame_count = defaultdict(int)
track_class_map   = {}
tracked_data      = defaultdict(list)
frame_buffer      = deque(maxlen=BUFFER_SIZE)
visual_id_map     = {}
next_visual_id    = defaultdict(lambda: 1)
vehicle_snapshots = {}
snapshot_placeholders = {}
best_vehicle_snapshots = {}
best_confidence_map = {}
plate_text_map = {} 
ocr_results = {}
fps_placeholder = st.sidebar.empty()
prev_time = time.time()

ocr_queue = queue.Queue()
ocr_lock = threading.Lock()
ocr_reader = easyocr.Reader(['en'], gpu=False)

best_vehicle_class_map = {} 

fuzzy_queue = queue.Queue()
fuzzy_results = {}  

FUZZY_THRESHOLD = 55  
CLIP_THRESHOLD = 0.856 

matched_queue = queue.Queue(maxsize=100)
matched_lock = threading.Lock()

matched_visual_ids = set()


#here image is converted to Blob  for uploading into the databse
def img_to_bytes(img):
    if img is None or img.size == 0:
        return None
    is_success, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes() if is_success else None

# fuzzy matching function for combination and matching with the datbase with proper threading and queuing
def fuzzy_worker():
    while True:
        visual_id, ocr_text, is_best = fuzzy_queue.get()
        if ocr_text is None:
            break  
        matches = process.extract(
            ocr_text, plate_db, scorer=fuzz.ratio, limit=3
        )
        with ocr_lock:
            fuzzy_results[(visual_id, is_best)] = matches  
        fuzzy_queue.task_done()

# OCR for extracting text from the number plate
def ocr_worker():
    while True:
        visual_id, plate_img, is_best = ocr_queue.get()
        if plate_img is None:
            break  
        result = ocr_reader.readtext(plate_img)
        text, conf = "", 0.0
        if result:
            text, conf = result[0][1], result[0][2]
        with ocr_lock:
            if visual_id not in ocr_results:
                ocr_results[visual_id] = {}
            ocr_results[visual_id]['best' if is_best else 'latest'] = (text, conf)
        ocr_queue.task_done()

        if text:  
            fuzzy_queue.put((visual_id, text, is_best))

ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
ocr_thread.start()

fuzzy_thread = threading.Thread(target=fuzzy_worker, daemon=True)
fuzzy_thread.start()

# function for retrieving number plates from the pre-existing database
def load_plate_numbers(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT number_plate FROM vehicles")
    plates = [row[0] for row in cursor.fetchall()]
    conn.close()
    return plates

plate_db = load_plate_numbers('D:/vehicles.db')


# function for retrieving other information
def load_plate_info(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT number_plate, vehicle_photo1, owner_name FROM vehicles")
    plate_info = {}
    for row in cursor.fetchall():
        plate, photo_blob, owner = row
        plate_info[plate] = {"vehicle_photo1": photo_blob, "owner_name": owner}
    conn.close()
    return plate_info

plate_info_map = load_plate_info('D:/vehicles.db')


# lodic code function for calculating overlapping
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea)

# function for detecting objects using YOLOv11 model
def detect_all(frame):
    detections = []
    models = [
        (vehicle_model, "vehicle", (0, 255, 0), CONF_THRESH["vehicle"]),
        (plate_model,   "plate",   (255, 0, 0), CONF_THRESH["plate"]),
        (face_model,    "face",    (0, 0, 255), CONF_THRESH["face"]),
    ]
    for model, label, color, conf_thresh in models:
        results = model(frame, verbose=False)[0]
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_idx = int(box.cls[0]) if hasattr(box, "cls") else -1
            detections.append([x1, y1, x2, y2, conf, label, color, class_idx])
    return detections

# for smoothing the tracking
def interpolate_buffer(data_buffer):
    interpolated_results = defaultdict(list)
    for track_id, entries in data_buffer.items():
        entries.sort()
        for i in range(len(entries) - 1):
            f1, box1 = entries[i]
            f2, box2 = entries[i + 1]
            interpolated_results[f1].append((track_id, box1, False))
            gap = f2 - f1
            if gap > 1:
                x = np.array([f1, f2])
                y = np.vstack([box1, box2])
                interp_func = interp1d(x, y, axis=0)
                for f in range(f1 + 1, f2):
                    interpolated_box = [int(b) for b in interp_func(f)]
                    interpolated_results[f].append((track_id, interpolated_box, True))
        interpolated_results[entries[-1][0]].append((track_id, entries[-1][1], False))
    return interpolated_results


# associate faces/plates to their respective vehicles 
def associate_entities_to_vehicles(vehicle_boxes, entities, mode="face"):
    associations = {}
    for e_id, e_box in entities.items():
        ex1, ey1, ex2, ey2 = e_box
        best_vehicle = None
        best_dist = float("inf")
        for v_id, v_box in vehicle_boxes.items():
            vx1, vy1, vx2, vy2 = v_box
            if ex1 >= vx1 and ey1 >= vy1 and ex2 <= vx2 and ey2 <= vy2:
                dist = abs(ey1 - vy1) if mode == "face" else abs(ey2 - vy2)
                if dist < best_dist:
                    best_dist = dist
                    best_vehicle = v_id
        if best_vehicle is not None:
            associations[e_id] = best_vehicle
    return associations


def create_matched_db(db_path='D:/detected_vehicles_log.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matched_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_name_db TEXT,
            vehicle_image_detected BLOB,
            face_image_detected BLOB,
            detected_number_plate TEXT,
            detection_timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

create_matched_db()

def matched_db_worker(db_path='D:/detected_vehicles_log.db'):
    while True:
        item = matched_queue.get()
        if item is None:
            break
        with matched_lock:
            vehicle_img_bytes = img_to_bytes(item['vehicle_image_detected'])
            face_img_bytes = img_to_bytes(item['face_image_detected'])
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO matched_vehicles (owner_name_db, vehicle_image_detected, face_image_detected, detected_number_plate, detection_timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (item['owner_name_db'], vehicle_img_bytes, face_img_bytes, item['detected_number_plate'], item['detection_timestamp']))
            conn.commit()
            conn.close()
        matched_queue.task_done()

threading.Thread(target=matched_db_worker, daemon=True).start()

# User Interface code
st.title("Live Interpolated Object Tracking")
frame_display = st.image([])
st.markdown("### 🚗 Vehicle Snapshots with Face & Plate & OCR")
snapshot_container = st.container()

if st.button("Run Detection"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Error opening video file.")
        st.stop()

    frame_number = 0

    start_image_similarity_thread()
    

    detected_vehicle_ids = set()
    # main loop for Display
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        detections = detect_all(frame)
        sort_input = np.array([[*det[:5]] for det in detections]) if detections else np.empty((0, 5))
        tracked = track_objects(sort_input.tolist())

        current_data = defaultdict(list)
        for trk in tracked:
            x1, y1, x2, y2, trk_id = map(int, trk[:5])
            best_iou = 0
            matched_det = None
            for det in detections:
                dx1, dy1, dx2, dy2 = det[:4]
                iou = compute_iou([x1, y1, x2, y2], [dx1, dy1, dx2, dy2])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    matched_det = det

            if matched_det:
                label = matched_det[5]
                color = matched_det[6]
                track_frame_count[trk_id] += 1
                thresholds = {"vehicle": 30, "number_plate": 0, "face": 30}
                required_frames = thresholds.get(label, 0)
                if track_frame_count[trk_id] >= required_frames and trk_id not in track_class_map:
                    track_class_map[trk_id] = (label, color)
                    visual_id_map[trk_id] = next_visual_id[label]
                   
                    class_name = "Unknown"
                    for det in detections:
                        dx1, dy1, dx2, dy2, _, det_label, _, class_idx = det
                        if det_label == "vehicle":
                            iou = compute_iou([x1, y1, x2, y2], [dx1, dy1, dx2, dy2])
                            if iou > 0.5:
                                class_name = vehicle_class_names.get(class_idx, "Unknown")
                                break
                    vehicle_class_map[visual_id_map[trk_id]] = class_name
                    detected_vehicle_ids.add(visual_id_map[trk_id]) 
                    next_visual_id[label] += 1

            current_data[trk_id].append((frame_number, [x1, y1, x2, y2]))

        for tid, entries in current_data.items():
            tracked_data[tid].extend(entries)

        frame_buffer.append((frame_number, frame.copy(), current_data))

        interp_target_frame = frame_number - INTERPOLATION_DELAY
        interp_frame = None
        interp_data = None
        for fn, fr, data in frame_buffer:
            if fn == interp_target_frame:
                interp_frame = fr
                interp_data = data
                break

        if interp_frame is not None and interp_data is not None:
            interpolated = interpolate_buffer(interp_data)
            draw_frame = interp_frame.copy()
            clean_snapshot = interp_frame.copy()

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                vehicles, faces, plates = {}, {}, {}

                for tid, box, is_interp in interpolated[interp_target_frame]:
                    if tid not in track_class_map or tid not in visual_id_map:
                        continue
                    label, color = track_class_map[tid]
                    visual_id = visual_id_map[tid]
                    x1, y1, x2, y2 = box

                    if label == "vehicle":
                        vehicles[tid] = box
                    elif label == "face":
                        faces[tid] = box
                    elif label == "plate":
                        plates[tid] = box

                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(draw_frame, f"{label} ID {visual_id} {'(I)' if is_interp else ''}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    writer.writerow([interp_target_frame, visual_id, f"{x1} {y1} {x2} {y2}", int(is_interp)])

                face_associations = associate_entities_to_vehicles(vehicles, faces, mode="face")
                plate_associations = associate_entities_to_vehicles(vehicles, plates, mode="plate")

                for face_id, vehicle_id in face_associations.items():
                    fx1, fy1, fx2, fy2 = faces[face_id]
                    fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                    vx1, vy1, vx2, vy2 = vehicles[vehicle_id]
                    vcx = (vx1 + vx2) // 2
                    cv2.line(draw_frame, (fcx, fcy), (vcx, vy1), (255, 0, 0), 2)

                for plate_id, vehicle_id in plate_associations.items():
                    px1, py1, px2, py2 = plates[plate_id]
                    pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                    vx1, vy1, vx2, vy2 = vehicles[vehicle_id]
                    vcx = (vx1 + vx2) // 2
                    cv2.line(draw_frame, (pcx, pcy), (vcx, vy2), (0, 0, 255), 2)

                with snapshot_container:
                    for tid, box in vehicles.items():
                        if tid in track_class_map and tid in visual_id_map:
                            label, _ = track_class_map[tid]
                            visual_id = visual_id_map[tid]
                            x1, y1, x2, y2 = box

                            vehicle_crop = clean_snapshot[y1:y2, x1:x2]
                            face_crop = None
                            plate_crop = None
                            plate_text = None

                            vehicle_conf = 0.0
                            for det in detections:
                                dx1, dy1, dx2, dy2, conf, det_label, _, class_idx = det
                                if det_label == "vehicle":
                                    iou = compute_iou([x1, y1, x2, y2], [dx1, dy1, dx2, dy2])
                                    if iou > 0.5:
                                        vehicle_conf = conf
                                        break

                            for fid, vid in face_associations.items():
                                if vid == tid and fid in faces:
                                    fx1, fy1, fx2, fy2 = faces[fid]
                                    face_crop = clean_snapshot[fy1:fy2, fx1:fx2]
                                    break

                            for pid, vid in plate_associations.items():
                                if vid == tid and pid in plates:
                                    px1, py1, px2, py2 = plates[pid]
                                    plate_crop = clean_snapshot[py1:py2, px1:px2]
                            # logic code for checking if the snapshot is best or not
                            if (visual_id not in best_confidence_map or
                                vehicle_conf > best_confidence_map[visual_id]):
                                prev_best = best_vehicle_snapshots.get(visual_id, {})
                                best_confidence_map[visual_id] = vehicle_conf
                                best_vehicle_snapshots[visual_id] = {
                                    "vehicle": vehicle_crop.copy(),
                                    "face": face_crop.copy() if face_crop is not None else prev_best.get("face"),
                                    "plate": plate_crop.copy() if plate_crop is not None else prev_best.get("plate")
                                }
                                                               
                                best_class_name = "Unknown"
                                for det in detections:
                                    dx1, dy1, dx2, dy2, conf, det_label, _, class_idx = det
                                    if det_label == "vehicle":
                                        iou = compute_iou([x1, y1, x2, y2], [dx1, dy1, dx2, dy2])
                                        if iou > 0.5:
                                            best_class_name = vehicle_class_names.get(class_idx)
                                            break
                                best_vehicle_class_map[visual_id] = best_class_name

                            else:
                                if visual_id in best_vehicle_snapshots:
                                    if face_crop is not None:
                                        best_vehicle_snapshots[visual_id]["face"] = face_crop.copy()
                                    if plate_crop is not None:
                                        best_vehicle_snapshots[visual_id]["plate"] = plate_crop.copy()

                            if visual_id not in snapshot_placeholders:
                                snapshot_placeholders[visual_id] = st.empty()

                            with snapshot_placeholders[visual_id].container():
                                st.markdown("---")  
                                st.markdown(f"### Vehicle ID {visual_id}")
                                vehicle_class = best_vehicle_class_map.get(visual_id)
                                st.markdown(f"**Class:** `{vehicle_class}`")
                                if plate_text_map.get(visual_id):
                                    st.markdown(f"**Plate OCR:** `{plate_text_map[visual_id]}`")

                                if visual_id in matched_visual_ids:
                                    st.markdown("**✅ Matched!**")
                                    best_snap = best_vehicle_snapshots.get(visual_id, {})
                                    bvc = best_snap.get("vehicle")
                                    bfc = best_snap.get("face")
                                    bpc = best_snap.get("plate")
                                    cols = st.columns(2)
                                    with cols[0]:
                                        if bvc is not None and bvc.size > 0:
                                            st.image(cv2.cvtColor(bvc, cv2.COLOR_BGR2RGB), caption="Vehicle", use_container_width=True)
                                        if bfc is not None and bfc.size > 0:
                                            st.image(cv2.cvtColor(bfc, cv2.COLOR_BGR2RGB), caption="Face", use_container_width=True)
                                        if bpc is not None and bpc.size > 0:
                                            st.image(cv2.cvtColor(bpc, cv2.COLOR_BGR2RGB), caption="Plate", use_container_width=True)
                                    with cols[1]:
                                        st.markdown("**Matched and saved to database.**")
                                else:
                                    st.markdown("**🚨 Suspicious: No match found in database!**")
                                    cols = st.columns(2)
                                    # here column which display latest or live snapshots of vehicle, face and number plate
                                    with cols[0]:
                                        st.markdown("#### Latest")
                                        if vehicle_crop.size > 0:
                                            st.image(cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB),
                                                     caption="Vehicle", use_container_width=True)
                                        if face_crop is not None and face_crop.size > 0:
                                            st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
                                                     caption="Face", use_container_width=True)
                                        if plate_crop is not None and plate_crop.size > 0:
                                            st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), caption="Plate", use_container_width=True)
                                            with ocr_lock:
                                                ocr_latest = ocr_results.get(visual_id, {}).get('latest', ("", 0.0))
                                                fuzzy_latest = fuzzy_results.get((visual_id, False), [])
                                            if ocr_latest[0]:
                                                st.markdown(f"**Plate OCR (Latest):** `{ocr_latest[0]}` (conf: {ocr_latest[1]:.2f})")
                                            else:
                                                st.markdown("**Plate OCR (Latest):** _No text detected_")
                                    # another column for best snapshots of Vehicle, face and number plate. it also contain top 3 fuzzy matched data
                                    with cols[1]:
                                        st.markdown("#### Best (Highest Confidence)")
                                        best_snap = best_vehicle_snapshots.get(visual_id, {})
                                        bvc = best_snap.get("vehicle")
                                        bfc = best_snap.get("face")
                                        bpc = best_snap.get("plate")
                                        if bvc is not None and bvc.size > 0:
                                            st.image(cv2.cvtColor(bvc, cv2.COLOR_BGR2RGB),
                                                     caption="Vehicle", use_container_width=True)
                                        if bfc is not None and bfc.size > 0:
                                            st.image(cv2.cvtColor(bfc, cv2.COLOR_BGR2RGB),
                                                     caption="Face", use_container_width=True)
                                        if bpc is not None and bpc.size > 0:
                                            st.image(cv2.cvtColor(bpc, cv2.COLOR_BGR2RGB), caption="Plate", use_container_width=True)
                                            with ocr_lock:
                                                ocr_best = ocr_results.get(visual_id, {}).get('best', ("", 0.0))
                                                fuzzy_best = fuzzy_results.get((visual_id, True), [])
                                            if fuzzy_best:
                                                st.markdown(f"**Plate OCR (best):** `{ocr_best[0]}` (conf: {ocr_best[1]:.2f})")
                                                st.markdown("**Top 3 Fuzzy Matches (Best):**")
                                                vehicle_matched = False
                                                matched_info = None
                                                for i, (match, score) in enumerate(fuzzy_best, 1):
                                                    info = plate_info_map.get(match)
                                                    cols = st.columns([2, 2, 3])
                                                    with cols[0]:
                                                        st.markdown(f"**{i}. `{match}`** (score: {score:.2f})")
                                                    with cols[1]:
                                                        if info and info['vehicle_photo1']:
                                                            try:
                                                                image = Image.open(BytesIO(info['vehicle_photo1']))
                                                                db_vehicle_img = np.array(image.convert("RGB"))
                                                                bvc = best_snap.get("vehicle")
                                                                if bvc is not None and bvc.size > 0:
                                                                    key = (visual_id, match)
                                                                    with image_similarity_lock:
                                                                        similarity = image_similarity_results.get(key)
                                                                    if similarity is None:
                                                                        queue_similarity_task(visual_id, match, bvc.copy(), db_vehicle_img.copy())
                                                                        st.markdown("**Image Similarity:** _Computing..._")
                                                                    else:
                                                                        st.markdown(f"**Image Similarity (CLIP cosine):** {similarity:.3f}")
                                                                        if score >= FUZZY_THRESHOLD and similarity >= CLIP_THRESHOLD and not vehicle_matched:
                                                                            st.markdown("**✅ MATCHED!**")
                                                                            vehicle_matched = True
                                                                            matched_info = {
                                                                                'owner_name': info['owner_name'],
                                                                                'plate': match,
                                                                                'db_vehicle_img': db_vehicle_img
                                                                            }
                                                                            if visual_id not in matched_visual_ids:
                                                                                try:
                                                                                    matched_queue.put_nowait({
                                                                                        'owner_name_db': info['owner_name'],
                                                                                        'vehicle_image_detected': best_snap.get("vehicle"),
                                                                                        'face_image_detected': best_snap.get("face"),
                                                                                        'detected_number_plate': match,
                                                                                        'detection_timestamp': datetime.datetime.now().isoformat()
                                                                                    })
                                                                                    matched_visual_ids.add(visual_id)
                                                                                except queue.Full:
                                                                                    pass
                                                                st.image(db_vehicle_img, caption="Vehicle Photo", use_container_width=True)
                                                            except Exception as e:
                                                                st.markdown(f"_Could not display vehicle photo: {e}_")
                                                        else:
                                                            st.markdown("_No vehicle photo available_")
                                                    with cols[2]:
                                                        if info:
                                                            st.markdown(f"**Owner Name:** {info['owner_name']}")
                                                        else:
                                                            st.markdown("_No additional info found in database_")
                                            else:
                                                st.markdown(f"**Plate OCR (best):** `{ocr_best[0]}` (conf: {ocr_best[1]:.2f})")
                                                st.markdown("**Top 3 Fuzzy Matches (Best):** _No matches found_")
                                                st.markdown("**Plate OCR (Best):** _No text detected_")
            # display frame rate per second
            frame_display.image(cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time
            fps_placeholder.markdown(f"**📸 FPS:** `{fps:.2f}`")
            time.sleep(0.01)

            # here it is the code for OCR by this EasyOCR perform OCR on number plate every 20 frames
            if frame_number % 20 == 0:
                for tid, box in vehicles.items():
                    if tid in visual_id_map:
                        visual_id = visual_id_map[tid]
                        plate_crop = None
                        for pid, vid in plate_associations.items():
                            if vid == tid and pid in plates:
                                px1, py1, px2, py2 = plates[pid]
                                plate_crop = clean_snapshot[py1:py2, px1:px2]
                                break
                        if plate_crop is not None and plate_crop.size > 0:
                            ocr_queue.put((visual_id, plate_crop.copy(), False))  

                        best_snap = best_vehicle_snapshots.get(visual_id, {})
                        bpc = best_snap.get("plate")
                        if bpc is not None and bpc.size > 0:
                            ocr_queue.put((visual_id, bpc.copy(), True)) 

        frame_number += 1

    cap.release()
    st.success(f"✅ Processing complete. Results saved to: {csv_path}")





