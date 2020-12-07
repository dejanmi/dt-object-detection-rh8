import numpy as np
import cv2
from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask

npz_index = 0

def save_npz(img, boxes, classes):
    global npz_index
    with makedirs("./data_collection/dataset"):
        np.savez(f"./data_collection/dataset/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1

def clean_segmented_image(seg_img):
    # TODO
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    boxes = []
    classes = []
    for obj, label in label_map.items():
        cu_mask = np.all(seg_img == label["color"], axis=-1)
        non_cu_mask = ~cu_mask
        mask_scaled_copy = seg_img.copy()
        mask_scaled_copy[cu_mask] = [255, 255, 255]
        mask_scaled_copy[non_cu_mask] = [0, 0, 0]
        mask_scaled_copy = cv2.medianBlur(mask_scaled_copy, 7)
        med_val = np.median(mask_scaled_copy)
        threshold1 = int(max(0, 0.7 * med_val))
        threshold2 = int(min(255, 1.3 * med_val))
        imgCanny = cv2.Canny(mask_scaled_copy, threshold1, threshold2)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        countours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in countours:
            area = cv2.contourArea(cnt)
            if area > 100:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                box = [x, y, x+w, y+h]
                boxes.append(box)
                classes.append(label["label"])
    return boxes, classes

seed(123)
environment = launch_env()
policy = PurePursuitPolicy(environment)
MAX_STEPS = 500
label_map = {"duckie": {"color": [100, 117, 226], "label": 1},
             "cone": {"color": [226, 111, 101], "label": 2},
             "truck": {"color": [116, 114, 117], "label": 3},
             "bus": {"color": [216, 171, 15], "label": 4}}

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []
    nb_of_steps = 0
    while True:
        action = policy.predict(np.array(obs))
        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array
        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)
        boxes, classes = clean_segmented_image(segmented_obs)
        if len(classes) > 0:
            save_npz(obs, boxes, classes)
        nb_of_steps += 1
        if done or nb_of_steps > MAX_STEPS:
            break