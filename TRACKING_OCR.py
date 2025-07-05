from sort import Sort

import numpy as np
import threading
from collections import defaultdict
import time

# Initialize tracker
tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)

# Tracking function
def track_objects(detections):
    sort_input = np.array(detections) if detections else np.empty((0, 5))
    tracked = tracker.update(sort_input)
    return tracked

# OCR function - run in background
