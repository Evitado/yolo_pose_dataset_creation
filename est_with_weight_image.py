from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

# ============================================================
# EDIT THESE PATHS
# ============================================================
MODEL_PATH = "/home/femi/yolo_pose_dataset_creation/runs/pose/aircraft_pose_new_clear_img10/weights/best.pt"
IMAGE_DIR = "/home/femi/yolo_pose_dataset_creation/aircraft_pose_with_normalising_applied_multifield_only_3/images/test"
OUTPUT_DIR = "/home/femi/yolo_pose_dataset_creation/predictions_labeled3"

# Confidence thresholds
BOX_CONF = 0.25
KPT_CONF = 0.25  # used only if keypoint conf is available

# ============================================================
# KEYPOINT NAMES
# IMPORTANT: order must match your dataset label order
# Example:
# 0 nose, 1 engine_left, 2 engine_right, 3 wing_left,
# 4 wing_right, 5 main_gear_left, 6 main_gear_right
# ============================================================
KEYPOINT_NAMES = [
    "N",   # nose
    "EL",  # engine left
    "ER",  # engine right
    # "WL",  # wing left
    # "WR",  # wing right
    # "ML",  # main gear left
    # "MR",  # main gear right
]

# ============================================================
# DRAW SETTINGS
# ============================================================
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
POINT_RADIUS = 4
TEXT_OFFSET_X = 6
TEXT_OFFSET_Y = -6
BOX_THICKNESS = 2


def draw_labeled_keypoints(image, result, image_name=""):
    """
    Draw detection boxes and labeled keypoints on the image.
    """
    img = image.copy()

    boxes = result.boxes
    keypoints = result.keypoints

    if boxes is None or len(boxes) == 0:
        cv2.putText(img, "No detections", (20, 40), FONT, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return img

    # Keypoint arrays
    kpts_xy = None
    kpts_conf = None

    if keypoints is not None:
        if keypoints.xy is not None:
            kpts_xy = keypoints.xy.cpu().numpy()   # shape: [num_det, num_kpts, 2]
        if keypoints.conf is not None:
            kpts_conf = keypoints.conf.cpu().numpy()  # shape: [num_det, num_kpts]

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    for det_idx, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)
        det_conf = float(confs[det_idx])

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICKNESS)

        # Box label
        box_text = f"aircraft {det_conf:.2f}"
        (tw, th), _ = cv2.getTextSize(box_text, FONT, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(img, box_text, (x1 + 3, y1 - 5), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw keypoints with names
        if kpts_xy is not None and det_idx < len(kpts_xy):
            for kp_idx, (x, y) in enumerate(kpts_xy[det_idx]):
                x = int(round(x))
                y = int(round(y))

                # Check confidence if available
                if kpts_conf is not None and det_idx < len(kpts_conf) and kp_idx < len(kpts_conf[det_idx]):
                    if float(kpts_conf[det_idx][kp_idx]) < KPT_CONF:
                        continue

                if kp_idx < len(KEYPOINT_NAMES):
                    kp_name = KEYPOINT_NAMES[kp_idx]
                else:
                    kp_name = f"K{kp_idx}"

                # Draw point
                cv2.circle(img, (x, y), POINT_RADIUS, (0, 0, 255), -1)

                # Draw text background
                text_x = x + TEXT_OFFSET_X
                text_y = y + TEXT_OFFSET_Y
                (txt_w, txt_h), _ = cv2.getTextSize(kp_name, FONT, FONT_SCALE, FONT_THICKNESS)

                bg_x1 = text_x - 2
                bg_y1 = text_y - txt_h - 2
                bg_x2 = text_x + txt_w + 2
                bg_y2 = text_y + 4

                cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                cv2.putText(img, kp_name, (text_x, text_y), FONT, FONT_SCALE, (255, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

    if image_name:
        cv2.putText(img, image_name, (20, img.shape[0] - 20), FONT, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    return img


def main():
    model = YOLO(MODEL_PATH)

    image_dir = Path(IMAGE_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in valid_exts])

    print(f"Found {len(image_paths)} images in: {image_dir}")

    for i, img_path in enumerate(image_paths, start=1):
        print(f"[{i}/{len(image_paths)}] Processing {img_path.name}")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Skipped unreadable image: {img_path}")
            continue

        results = model.predict(
            source=str(img_path),
            conf=BOX_CONF,
            verbose=False
        )

        if len(results) == 0:
            print("  No result returned by model.")
            continue

        labeled = draw_labeled_keypoints(image, results[0], image_name=img_path.name)

        save_path = output_dir / img_path.name
        ok = cv2.imwrite(str(save_path), labeled)
        if not ok:
            print(f"  Failed to save: {save_path}")
        else:
            print(f"  Saved: {save_path}")

    print(f"\nDone. Labeled images saved to:\n{output_dir}")


if __name__ == "__main__":
    main()