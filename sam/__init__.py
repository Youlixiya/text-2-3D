import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
device = "cuda" if torch.cuda.is_available() else "cpu"
#sam
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def load_segment_anything(sam_checkpoint, model_type):
    sam = sam_model_registry[model_type](checkpoint=f'pretrained/sam/{sam_checkpoint}')
    sam.to(device)
    global sam_predictor
    sam_predictor = SamPredictor(sam)
    return 'sam loads!'

def clear_sam():
    global sam_predictor
    del sam_predictor
    torch.cuda.empty_cache()
    return 'sam clears!'

def add_points_boxes(image, image_name, input_point_x, input_point_y, input_label, input_boxes_x1, input_boxes_y1, input_boxes_x2, input_boxes_y2):
    dpi=100
    plt.figure(figsize=(image.shape[1]/dpi,image.shape[0]/dpi), dpi=dpi)
    plt.imshow(image)
    if input_point_x and input_point_y and input_label:
        point_x, point_y, input_label = input_point_x.split(','), input_point_y.split(','), input_label.split(',')
        points = np.zeros((len(point_x), 2), dtype=np.int64)
        labels = np.zeros(len(point_x), dtype=np.int64)
        for i in range(len(point_x)):
            points[i, 0] = int(point_x[i])
            points[i, 1] = int(point_y[i])
            labels[i] = int(input_label[i])
        show_points(points, labels, plt.gca())
    if input_boxes_x1 and input_boxes_y1 and input_boxes_x2 and input_boxes_y2:
        boxes_x1, boxes_y1, boxes_x2, boxes_y2 = input_boxes_x1.split(','), input_boxes_y1.split(','), input_boxes_x2.split(','), input_boxes_y2.split(',')
        boxes = np.zeros((4), dtype=np.int64)
        for i in range(len(boxes_x1)):
            boxes[0] = int(boxes_x1[i])
            boxes[1] = int(boxes_y1[i])
            boxes[2] = int(boxes_x2[i])
            boxes[3] = int(boxes_y2[i])
        show_box(boxes, plt.gca())
    plt.axis('off')
    plt.savefig(f'images/{image_name}_points_boxes.jpg', bbox_inches='tight', pad_inches = -0.1, dpi=dpi)
    return cv2.cvtColor(cv2.imread(f'images/{image_name}_points_boxes.jpg'), cv2.COLOR_BGR2RGB)
def segment(image, image_name, input_point_x, input_point_y, input_label, input_boxes_x1, input_boxes_y1, input_boxes_x2, input_boxes_y2, segment_index, return_object = False):
    dpi=100
    plt.figure(figsize=(image.shape[1]/dpi,image.shape[0]/dpi), dpi=dpi)
    plt.imshow(image)
    sam_predictor.set_image(image)
    if isinstance(return_object, str):
        return_object = eval(return_object)
    if input_point_x and input_point_y and input_label:
        point_x, point_y, input_label = input_point_x.split(','), input_point_y.split(','), input_label.split(',')
        points = np.zeros((len(point_x), 2), dtype=np.int64)
        labels = np.zeros(len(point_x), dtype=np.int64)
        for i in range(len(point_x)):
            points[i, 0] = int(point_x[i])
            points[i, 1] = int(point_y[i])
            labels[i] = int(input_label[i])
        show_points(points, labels, plt.gca())
    if input_boxes_x1 and input_boxes_y1 and input_boxes_x2 and input_boxes_y2:
        boxes_x1, boxes_y1, boxes_x2, boxes_y2 = input_boxes_x1.split(','), input_boxes_y1.split(','), input_boxes_x2.split(','), input_boxes_y2.split(',')
        boxes = np.zeros((4), dtype=np.int64)
        for i in range(len(boxes_x1)):
            boxes[0] = int(boxes_x1[i])
            boxes[1] = int(boxes_y1[i])
            boxes[2] = int(boxes_x2[i])
            boxes[3] = int(boxes_y2[i])
        show_box(boxes, plt.gca())
    if input_point_x and input_point_y and input_label:
        if input_boxes_x1 and input_boxes_y1 and input_boxes_x2 and input_boxes_y2:
            masks, scores, logits = sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                box = boxes,
                multimask_output=True)
        else:
            masks, scores, logits = sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True)
    index = int(segment_index.split('_')[1])
    if return_object:
        object_rgb = image * (masks[index].reshape(image.shape[0], image.shape[1], 1))
        object_rgb = object_rgb.astype(np.uint8)
        bkgd_mask = np.where(object_rgb == 0, 1, 0)
        bkgd_mask *= 255
        bkgd_mask = bkgd_mask.astype(np.uint8)
        object_rgb += bkgd_mask
        cv2.imwrite(f'images/{image_name}_object_{index+1}.png', object_rgb[:,:,::-1])
        return object_rgb
    show_mask(masks[index], plt.gca())
    show_points(points, labels, plt.gca())
    plt.title(f"Mask {index+1}, Score: {scores[i]:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(f'images/{image_name}_segment_{index+1}.png', bbox_inches='tight', dpi=dpi)
    return cv2.cvtColor(cv2.imread(f'images/{image_name}_segment_{index+1}.png'), cv2.COLOR_BGR2RGB)