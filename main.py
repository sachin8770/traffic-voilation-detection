from ultralytics import YOLO
import cv2
import numpy as np
import os

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import requests
import glob
import cloudinary
import cloudinary.uploader

from plate_tracker import PlateTracker

API_URL = "http://127.0.0.1:3000/api/plates"

cloudinary.config(
  cloud_name = "dozg3fcta",
  api_key = "161944285762788",
  api_secret = "GzhRJ-3Hvl_1ETsp0_pVnIgKt2Q"
)

sent_to_api_ids = set()

def send_to_api(car_id, text, confidence, num_readings):
    car_id = int(car_id)
    if car_id in sent_to_api_ids:
        return  # already sent for this car
    sent_to_api_ids.add(car_id)

    image_url = None
    possible_images = glob.glob(f"./violations/violation_car_{car_id}_frame_*.jpg")
    if possible_images:
        snap_filename = possible_images[0]
        try:
            print(f"Uploading {snap_filename} to Cloudinary...")
            res = cloudinary.uploader.upload(snap_filename)
            image_url = res.get("secure_url")
            print(f"Upload successful: {image_url}")
        except Exception as e:
            print(f"Cloudinary upload failed: {e}")

    try:
        requests.post(API_URL, json={
            "car_id": car_id,
            "plate_text": text,
            "confidence": round(confidence, 4) if confidence else 0.0,
            "num_readings": num_readings,
            "video_source": "sample.mp4",
            "image_url": image_url
        }, timeout=2)
    except requests.RequestException as e:
        print(f"[api] failed to send car {car_id}: {e}")
    else:
        print(f"[DEBUG] POST sent for car {car_id}, plate={text}")

results = {}
tracker = PlateTracker(on_finalize=send_to_api, min_readings=2)

mot_tracker = Sort()

# load models
# yolov8n is the smallest/fastest model but misses small, distant vehicles
# (like ones still above a far-off violation line). Use 's' or 'm' for
# better range if you have the GPU/CPU budget for it.
coco_model = YOLO('yolov8s.pt')
DETECTION_CONF = 0.15  # lower than default (~0.25) to catch small/distant vehicles
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
video_path = 'Recording 2026-08-30 112842.mp4'
cap = cv2.VideoCapture(video_path)

# Create resizable live video window
cv2.namedWindow('Traffic Violation Detection - Live', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Traffic Violation Detection - Live', 1024, 768)

vehicles = [2, 3, 5, 7]

# Set of vehicle IDs that committed a violation (crossed line)
violating_vehicles = set()

# Track each vehicle's previous frame center-y position, so we can detect
# an actual crossing (was above the line, now at/below it) instead of a
# one-shot static position check.
prev_positions = {}

# Track the highest-scoring license plate reading for each vehicle
best_plates = {}

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # Define violation line position at 60% of frame height
        frame_h, frame_w = frame.shape[:2]
        line_y = int(frame_h * 0.70)
        # Draw Red Violation Line across frame
        cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 0, 255), 3)
        cv2.putText(frame, "VIOLATION LINE", (20, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # detect vehicles
        detections = coco_model(frame, verbose=False, conf=DETECTION_CONF)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))
        
        current_ids = [int(t[4]) for t in track_ids]
        tracker.check_lost_tracks(current_ids)

        # Collect IDs seen this frame so we can prune stale entries from
        # prev_positions (cars that left the frame / lost track).
        seen_this_frame = set()

        # Check line crossing & draw tracked vehicle bounding boxes
        for track in track_ids:
            x1, y1, x2, y2, car_id = track
            car_id = int(car_id)
            seen_this_frame.add(car_id)

            # Use the center of the box, not the bottom edge — the bottom
            # edge is almost always past a high-up line the instant a tall
            # vehicle box appears, which was causing instant false violations.
            cy = (y1 + y2) / 2.0

            prev_cy = prev_positions.get(car_id)
            # Flag a violation on ANY crossing, regardless of direction of
            # travel (top->bottom or bottom->top) — don't assume which way
            # traffic moves in this particular clip.
            if prev_cy is not None:
                was_above = prev_cy < line_y
                is_above = cy < line_y
                if was_above != is_above:
                    if car_id not in violating_vehicles:
                        violating_vehicles.add(car_id)
                        
                        # Take screenshot of violation and save to file
                        os.makedirs('./violations', exist_ok=True)
                        snap_filename = f'./violations/violation_car_{car_id}_frame_{frame_nmr}.jpg'
                        
                        snap_frame = frame.copy()
                        cv2.rectangle(snap_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        cv2.putText(snap_frame, f"VIOLATION - Vehicle ID: {car_id}", (int(x1), max(25, int(y1) - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.line(snap_frame, (0, line_y), (frame_w, line_y), (0, 0, 255), 3)
                        cv2.imwrite(snap_filename, snap_frame)
                        print(f"-> [VIOLATION ALERT] Screenshot saved: {snap_filename}", flush=True)

                        # Send to API immediately using whatever plate reading exists so far
                        # (may be None yet — plate is often read on a later frame)
                        plate_info = best_plates.get(car_id)
                        send_to_api(
                            car_id,
                            plate_info['text'] if plate_info else None,
                            plate_info['score'] if plate_info else 0.0,
                            1 if plate_info else 0,
                        )

            prev_positions[car_id] = cy

            # DEBUG: print each vehicle's center-y vs the line so you can see
            # whether vehicles ever actually approach/cross line_y in this
            # video, and recalibrate line_y (currently 0.40) if not.
            if frame_nmr % 15 == 0:
                print(f"[debug] frame {frame_nmr} car {car_id}: cy={cy:.0f} line_y={line_y}", flush=True)

            # Highlight violating vehicles in RED, normal vehicles in GREEN
            if car_id in violating_vehicles:
                box_color = (0, 0, 255)  # Red
                label = f"VIOLATION - ID: {car_id}"
            else:
                box_color = (0, 255, 0)  # Green
                label = f"Vehicle ID: {car_id}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.putText(frame, label, (int(x1), max(20, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Drop tracking history for IDs that vanished this frame (left the
        # scene or were dropped by SORT), so stale positions don't linger.
        for stale_id in list(prev_positions.keys()):
            if stale_id not in seen_this_frame:
                del prev_positions[stale_id]

        # Display total violation count HUD overlay
        cv2.rectangle(frame, (10, 10), (380, 55), (0, 0, 0), -1)
        cv2.putText(frame, f"TOTAL VIOLATIONS: {len(violating_vehicles)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # detect license plates
        license_plates = license_plate_detector(frame, verbose=False)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Draw license plate bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                if license_plate_crop.size > 0:
                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                        tracker.add_reading(car_id, license_plate_text, license_plate_text_score, score)

                        # Update the best plate if this new score is higher
                        if car_id not in best_plates or license_plate_text_score > best_plates[car_id]['score']:
                            best_plates[car_id] = {'text': license_plate_text, 'score': license_plate_text_score}

                        # Always display the best reading we've seen so far
                        best_text = best_plates[car_id]['text']
                        best_score = best_plates[car_id]['score']

                        # Display license plate text on live feed & terminal
                        cv2.putText(frame, f"Plate: {best_text}", (int(x1), max(20, int(y1) - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        print(f"Frame {frame_nmr} | Vehicle #{int(car_id)} | Plate (Best): {best_text} (Score: {best_score:.2f})", flush=True)

        # Render live frame window
        cv2.imshow('Traffic Violation Detection - Live', frame)

        # Press 'q' to stop live stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Live stream stopped by user.", flush=True)
            break

cap.release()
cv2.destroyAllWindows()

tracker.finalize_all()
# write results to CSV
write_csv(results, './test.csv')
print("Detection results successfully saved to test.csv", flush=True)
