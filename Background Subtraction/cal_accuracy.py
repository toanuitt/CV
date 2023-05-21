import cv2
import math
import glob
import numpy as np

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0


    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 50:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)

    intersection_area = max(0, abs(x_inter2 - x_inter1 + 1)) * max(0, abs(y_inter2 - y_inter1 + 1))

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


if __name__ == "__main__":
    object_detector = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=100)

    tracker = EuclideanDistTracker()

    bounding_box_file = "MOT15/train/KITTI-13/gt/gt.txt"
    image_folder = "MOT15/train/KITTI-13/img1/"
    image_files = glob.glob(image_folder + "*.jpg")
    image_count = len(image_files)
    with open(bounding_box_file, "r") as file:
        lines = file.readlines()

    true_positive = 0
    false_positive = 0
    false_negative = 0
    for line in lines:
        data = line.strip().split(",")
        frame_index = int(data[0])
        object_id = int(data[1])
        x = int(float(data[2]))
        y = int(float(data[3]))
        width = int(float(data[4]))
        height = int(float(data[5]))
        confidence = float(data[6])
        x_velocity = float(data[7])
        y_velocity = float(data[8])
        class_label = data[9]

        image_path = image_folder + str(frame_index).zfill(6) + ".jpg"
        frame = cv2.imread(image_path)

        mask = object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

        boxes_ids = tracker.update(detections)
        is_detected = False
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            iou = calculate_iou([x, y, x + w, y + h], [x, y, x + width, y + height])
            if iou > 0.5:
                true_positive += 1
                is_detected = True
                break

        if not is_detected:
            false_negative += 1

        if len(boxes_ids) > 0 and not is_detected:
            false_positive += 1

    accuracy = true_positive / (true_positive + false_positive + false_negative)
    print("Accuracy:", accuracy)
    print(image_count)
