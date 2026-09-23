import cv2
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO
import timm
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_FILE = "data/vision/sample.mp4"

# YOLO face detector
YOLO_MODEL = "vision/yolov11n-face.pt"

# Existing HSEmotion EfficientNet checkpoint
EMOTION_MODEL = (
    Path.home()
    / ".hsemotion"
    / "enet_b0_8_best_afew.pt"
)

DEVICE = "cpu"

# Analyze every Nth frame
FRAME_INTERVAL = 10

# Minimum YOLO face confidence
FACE_CONFIDENCE = 0.40

# IoU threshold used for tracking
IOU_THRESHOLD = 0.30

# Maximum number of frames a track can disappear
MAX_MISSED_FRAMES = 5


# ============================================================
# EMOTION LABELS
# ============================================================

EMOTION_LABELS = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]


# ============================================================
# LOAD YOLO
# ============================================================

def load_face_detector():

    print("=" * 60)
    print("LOADING YOLO FACE DETECTOR")
    print("=" * 60)

    print("Model:", YOLO_MODEL)

    model = YOLO(YOLO_MODEL)

    print("YOLO face detector loaded.")

    return model


# ============================================================
# LOAD EFFICIENTNET
# ============================================================

def load_emotion_model():

    print("=" * 60)
    print("LOADING EFFICIENTNET-B0 EMOTION MODEL")
    print("=" * 60)

    print("Model:", EMOTION_MODEL)

    if not EMOTION_MODEL.exists():
        raise FileNotFoundError(
            f"Emotion model not found:\n{EMOTION_MODEL}"
        )

    # IMPORTANT:
    # weights_only=False is required because this checkpoint
    # contains the complete EfficientNet model object.
    model = torch.load(
        EMOTION_MODEL,
        map_location=DEVICE,
        weights_only=False
    )

    print("Loaded model:", type(model))
    print("Classifier:", model.classifier)

    model = model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("EfficientNet feature extractor ready.")
    print("Feature dimension: 1280")
    print("Number of emotions: 8")

    return model, transform


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width * intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# FACE TRACKER
# ============================================================

class FaceTracker:

    def __init__(
        self,
        iou_threshold=IOU_THRESHOLD,
        max_missed_frames=MAX_MISSED_FRAMES
    ):

        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames

        self.next_track_id = 1

        # {
        #   track_id: {
        #       "bbox": [...],
        #       "missed": 0,
        #       "emotions": [...]
        #   }
        # }
        self.tracks = {}

    # --------------------------------------------------------
    # UPDATE TRACKS
    # --------------------------------------------------------

    def update(self, detections):

        """
        detections:

        [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.95
            }
        ]
        """

        matched_tracks = set()
        matched_detections = set()

        assignments = []

        # ----------------------------------------------------
        # Match existing tracks with current detections
        # ----------------------------------------------------

        for track_id, track in list(self.tracks.items()):

            best_iou = 0.0
            best_detection = None

            for detection_index, detection in enumerate(
                detections
            ):

                if detection_index in matched_detections:
                    continue

                iou = calculate_iou(
                    track["bbox"],
                    detection["bbox"]
                )

                if iou > best_iou:

                    best_iou = iou
                    best_detection = detection_index

            # ------------------------------------------------
            # Match found
            # ------------------------------------------------

            if (
                best_detection is not None
                and best_iou >= self.iou_threshold
            ):

                detection = detections[best_detection]

                track["bbox"] = detection["bbox"]
                track["missed"] = 0

                matched_tracks.add(track_id)
                matched_detections.add(best_detection)

                assignments.append(
                    (
                        track_id,
                        detection
                    )
                )

        # ----------------------------------------------------
        # Create new tracks
        # ----------------------------------------------------

        for detection_index, detection in enumerate(
            detections
        ):

            if detection_index in matched_detections:
                continue

            track_id = self.next_track_id

            self.next_track_id += 1

            self.tracks[track_id] = {
                "bbox": detection["bbox"],
                "missed": 0,
                "emotions": []
            }

            assignments.append(
                (
                    track_id,
                    detection
                )
            )

        # ----------------------------------------------------
        # Increase missed count
        # ----------------------------------------------------

        for track_id in list(self.tracks.keys()):

            if track_id not in matched_tracks:

                # New tracks are already considered active
                is_new_track = (
                    track_id >= self.next_track_id - len(
                        detections
                    )
                )

                if not is_new_track:

                    self.tracks[track_id][
                        "missed"
                    ] += 1

        # ----------------------------------------------------
        # Remove old tracks
        # ----------------------------------------------------

        for track_id in list(self.tracks.keys()):

            if (
                self.tracks[track_id]["missed"]
                > self.max_missed_frames
            ):

                del self.tracks[track_id]

        return assignments


# ============================================================
# EMOTION PREDICTION
# ============================================================

def predict_emotion(
    face_image,
    emotion_model,
    transform
):

    image = Image.fromarray(face_image)

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        output = emotion_model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    probabilities = probabilities[0].cpu().numpy()

    emotion_index = int(
        np.argmax(probabilities)
    )

    emotion = EMOTION_LABELS[
        emotion_index
    ]

    confidence = float(
        probabilities[emotion_index]
    )

    return emotion, confidence, probabilities


# ============================================================
# DRAW RESULT
# ============================================================

def draw_result(
    frame,
    bbox,
    track_id,
    emotion,
    confidence
):

    x1, y1, x2, y2 = map(
        int,
        bbox
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = (
        f"ID {track_id} | "
        f"{emotion} "
        f"{confidence:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(video_file):

    print("\n" + "=" * 60)
    print("FACIAL EMOTION ANALYSIS")
    print("=" * 60)

    face_detector = load_face_detector()

    emotion_model, transform = (
        load_emotion_model()
    )

    # --------------------------------------------------------
    # Tracker
    # --------------------------------------------------------

    tracker = FaceTracker()

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_file
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Unable to open video: {video_file}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print("\nVideo information")
    print("------------------")
    print("Total frames:", total_frames)
    print("FPS:", fps)

    # --------------------------------------------------------
    # Per-person emotion history
    # --------------------------------------------------------

    person_emotions = defaultdict(list)

    frame_count = 0
    analyzed_frames = 0
    total_faces = 0

    # --------------------------------------------------------
    # Video loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # ----------------------------------------------------
        # Analyze every Nth frame
        # ----------------------------------------------------

        if frame_count % FRAME_INTERVAL != 0:
            continue

        analyzed_frames += 1

        # ----------------------------------------------------
        # YOLO face detection
        # ----------------------------------------------------

        results = face_detector(
            frame,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                if confidence < FACE_CONFIDENCE:
                    continue

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                detections.append({
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ],
                    "confidence": confidence
                })

        total_faces += len(detections)

        # ----------------------------------------------------
        # TRACK FACES
        # ----------------------------------------------------

        assignments = tracker.update(
            detections
        )

        # ----------------------------------------------------
        # Emotion for each tracked face
        # ----------------------------------------------------

        for track_id, detection in assignments:

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            # Clamp coordinates
            h, w = frame.shape[:2]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))

            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            # ------------------------------------------------
            # Crop face
            # ------------------------------------------------

            face = frame[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            # BGR -> RGB
            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # Emotion
            # ------------------------------------------------

            emotion, confidence, probabilities = (
                predict_emotion(
                    face_rgb,
                    emotion_model,
                    transform
                )
            )

            # ------------------------------------------------
            # Store result for this PERSON
            # ------------------------------------------------

            person_emotions[
                track_id
            ].append({
                "frame": frame_count,
                "emotion": emotion,
                "confidence": confidence
            })

            # ------------------------------------------------
            # Save emotion in tracker
            # ------------------------------------------------

            if track_id in tracker.tracks:

                tracker.tracks[
                    track_id
                ]["emotions"].append(
                    emotion
                )

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print(
                f"Frame {frame_count:5d} | "
                f"Person {track_id:2d} | "
                f"{emotion:10s} | "
                f"{confidence:.4f}"
            )

            # ------------------------------------------------
            # Draw
            # ------------------------------------------------

            draw_result(
                frame,
                detection["bbox"],
                track_id,
                emotion,
                confidence
            )

    cap.release()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("VIDEO EMOTION RESULT")
    print("=" * 60)

    print("Total frames     :", total_frames)
    print("Frames analyzed  :", analyzed_frames)
    print("Faces detected   :", total_faces)
    print("Persons tracked  :", len(person_emotions))

    # ========================================================
    # PER PERSON RESULTS
    # ========================================================

    for person_id, records in person_emotions.items():

        print("\n" + "-" * 60)
        print(f"PERSON {person_id}")
        print("-" * 60)

        emotion_counts = defaultdict(int)
        confidence_values = defaultdict(list)

        for record in records:

            emotion = record["emotion"]
            confidence = record["confidence"]

            emotion_counts[emotion] += 1

            confidence_values[
                emotion
            ].append(confidence)

        total = len(records)

        print(
            "Emotion distribution"
        )

        print("--------------------")

        for emotion, count in sorted(
            emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            percentage = (
                count / total * 100
            )

            avg_confidence = (
                sum(
                    confidence_values[
                        emotion
                    ]
                )
                / len(
                    confidence_values[
                        emotion
                    ]
                )
            )

            print(
                f"{emotion:10s} : "
                f"{count:4d} "
                f"({percentage:6.2f}%) "
                f"avg_conf={avg_confidence:.4f}"
            )

        # ----------------------------------------------------
        # Dominant emotion
        # ----------------------------------------------------

        dominant_emotion = max(
            emotion_counts,
            key=emotion_counts.get
        )

        dominant_count = (
            emotion_counts[
                dominant_emotion
            ]
        )

        dominant_confidence = (
            sum(
                confidence_values[
                    dominant_emotion
                ]
            )
            /
            len(
                confidence_values[
                    dominant_emotion
                ]
            )
        )

        print()

        print(
            "Dominant emotion :",
            dominant_emotion
        )

        print(
            "Dominant count   :",
            dominant_count
        )

        print(
            "Average confidence:",
            f"{dominant_confidence:.4f}"
        )

    print("\n" + "=" * 60)

    return person_emotions


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    analyze_video(
        VIDEO_FILE
    )

