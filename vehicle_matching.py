import cv2
import numpy as np
import torch
import clip
from PIL import Image
import threading
import queue
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s:%(message)s')

clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")

image_similarity_queue = queue.Queue()
image_similarity_results = {}
image_similarity_lock = threading.Lock()
_similarity_thread_started = False
_similarity_queued_set = set()

def get_clip_embedding(img):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Input image is invalid or empty.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    image_input = clip_preprocess(pil_img).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()[0]

def clip_cosine_similarity(emb1, emb2):
    return float(np.dot(emb1, emb2))

def image_similarity_worker():
    while True:
        visual_id, match_plate, best_vehicle_img, db_vehicle_img = image_similarity_queue.get()
        if best_vehicle_img is None or db_vehicle_img is None:
            image_similarity_queue.task_done()
            continue
        try:
            # Ensure both images are valid and have 3 channels
            if best_vehicle_img.ndim == 2:
                best_vehicle_img = cv2.cvtColor(best_vehicle_img, cv2.COLOR_GRAY2BGR)
            if db_vehicle_img.ndim == 2:
                db_vehicle_img = cv2.cvtColor(db_vehicle_img, cv2.COLOR_GRAY2BGR)
            if best_vehicle_img.shape[2] == 4:
                best_vehicle_img = cv2.cvtColor(best_vehicle_img, cv2.COLOR_BGRA2BGR)
            if db_vehicle_img.shape[2] == 4:
                db_vehicle_img = cv2.cvtColor(db_vehicle_img, cv2.COLOR_BGRA2BGR)
            db_vehicle_img_resized = cv2.resize(db_vehicle_img, (best_vehicle_img.shape[1], best_vehicle_img.shape[0]))
            emb1 = get_clip_embedding(best_vehicle_img)
            emb2 = get_clip_embedding(db_vehicle_img_resized)
            similarity = clip_cosine_similarity(emb1, emb2)
        except Exception as e:
            similarity = 0.0
        with image_similarity_lock:
            image_similarity_results[(visual_id, match_plate)] = similarity
            _similarity_queued_set.discard((visual_id, match_plate))
        image_similarity_queue.task_done()

def start_image_similarity_thread():
    global _similarity_thread_started
    if not _similarity_thread_started:
        thread = threading.Thread(target=image_similarity_worker, daemon=True)
        thread.start()
        _similarity_thread_started = True

def is_similarity_computed_or_queued(visual_id, match_plate):
    with image_similarity_lock:
        if (visual_id, match_plate) in image_similarity_results:
            return True
        if (visual_id, match_plate) in _similarity_queued_set:
            return True
        return False

def queue_similarity_task(visual_id, match_plate, best_vehicle_img, db_vehicle_img):
    with image_similarity_lock:
        key = (visual_id, match_plate)
        if key in image_similarity_results or key in _similarity_queued_set:
            return
        _similarity_queued_set.add(key)
    image_similarity_queue.put((visual_id, match_plate, best_vehicle_img, db_vehicle_img))