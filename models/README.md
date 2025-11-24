
# YOLO-Pose Aircraft Model Weights

This folder contains the trained YOLOv8-Pose model for aircraft keypoint estimation

---

## Training Command Used

```
yolo pose train \
    model=yolov8s-pose.pt \
    data=./aircraft_pose_all/aircraft_pose.yaml \
    imgsz=1024 \
    epochs=150 \
    batch=8 \
    device=0 \
    verbose=True
```

---

## Dataset Structure

The dataset used for training was created using the dataset exporter with synthetic keypoints and visibility filtering.

```
aircraft_pose_all/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  aircraft_pose.yaml
```

---

## Trained Weights

The trained YOLOv8-Pose weights are stored here:

```
models/aircraft_pose_yolov8s_best.pt
```

---


