"""
ViLD - Vision and Language Knowledge Distillation for Open-Vocabulary Object Detection.

This module provides the ViLD (Vision-Language Detection) model for open-vocabulary
object detection. ViLD enables detecting objects beyond a fixed set of categories
by leveraging CLIP embeddings.

Key Components:
- Text embedding building with prompt engineering
- Object detection with confidence scoring
- Visualization of detection results

Original ViLD Repository:
    https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild

Reference:
    Gu, X., Lin, T., Kuo, C., & Cui, Y. (2021). Open-Vocabulary Object Detection
    via Vision and Language Knowledge Distillation. arXiv preprint arXiv:2104.13921.

Used in SayCan:
    https://github.com/google-research/google-research/tree/master/saycan
"""

import os
import collections
import numpy as np
import cv2
import torch
import clip
import matplotlib.pyplot as plt
from tqdm import tqdm
from easydict import EasyDict
from PIL import Image
import tensorflow.compat.v1 as tf

# Get the directory where this script is located
SAYCAN_DIR = os.path.dirname(os.path.abspath(__file__))


def softmax(x, axis=-1):
    """Compute softmax values for each element in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ViLD configuration flags
FLAGS = {
    'prompt_engineering': True,
    'this_is': True,
    'temperature': 100.0,
    'use_softmax': False,
}
FLAGS = EasyDict(FLAGS)

# Visualization parameters
display_input_size = (10, 10)
overall_fig_size = (18, 24)
line_thickness = 1
fig_size_w = 35
mask_color = 'red'
alpha = 0.5


def article(name):
    """Return 'an' if name starts with a vowel, 'a' otherwise."""
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    """Process category name by replacing underscores and slashes."""
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


# Prompt templates for CLIP embedding
single_template = ["a photo of {article} {}."]

multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',
    'itap of {article} {}.',
    'itap of my {}.',
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',
    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',
    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',
    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',
    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',
    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',
    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',
    'a plastic {}.',
    'the plastic {}.',
    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',
    'an embroidered {}.',
    'the embroidered {}.',
    'a painting of the {}.',
    'a painting of a {}.',
]

# Load CLIP model
clip_model, clip_preprocess = clip.load("ViT-B/32")
if torch.cuda.is_available():
    clip_model.cuda()
clip_model.eval()
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
print("Input resolution:", clip_model.visual.input_resolution)
print("Context length:", clip_model.context_length)
print("Vocab size:", clip_model.vocab_size)


def build_text_embedding(categories):
    """
    Build text embeddings for object categories using CLIP.

    Args:
        categories: List of category dicts with 'name' and 'id' keys

    Returns:
        Numpy array of text embeddings
    """
    if FLAGS.prompt_engineering:
        templates = multiple_templates
    else:
        templates = single_template

    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        all_text_embeddings = []
        print("Building text embeddings...")
        for category in tqdm(categories):
            texts = [
                template.format(processed_name(category["name"], rm_dot=True),
                                article=article(category["name"]))
                for template in templates
            ]
            if FLAGS.this_is:
                texts = [
                    "This is " + text if text.startswith("a") or text.startswith("the") else text
                    for text in texts
                ]
            texts = clip.tokenize(texts)
            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = clip_model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()
    return all_text_embeddings.cpu().numpy().T


# Load ViLD TensorFlow model
config = tf.ConfigProto(allow_soft_placement=True)
if torch.cuda.is_available():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
session = tf.Session(graph=tf.Graph(), config=config)
saved_model_dir = os.path.join(SAYCAN_DIR, "image_path_v2")
_ = tf.saved_model.loader.load(session, ["serve"], saved_model_dir)

numbered_categories = [{"name": str(idx), "id": idx} for idx in range(50)]
numbered_category_indices = {cat["id"]: cat for cat in numbered_categories}


def nms(dets, scores, thresh, max_dets=1000):
    """
    Non-maximum suppression.

    Args:
        dets: Detection boxes [N, 4]
        scores: Detection scores [N,]
        thresh: IoU threshold
        max_dets: Maximum detections to keep

    Returns:
        List of indices to keep
    """
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0 and len(keep) < max_dets:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        overlap = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-12)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]
    return keep


import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = ["White"]


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color="red", thickness=4,
                                display_str_list=(), use_normalized_coordinates=True):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (
            xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height
        )
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
              width=thickness, fill=color)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    display_str_heights = [font.getbbox(ds)[3] - font.getbbox(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    for display_str in display_str_list[::-1]:
        text_left = min(5, left)
        bbox = font.getbbox(display_str)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str,
                  fill="black", font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, color="red", thickness=4,
                                      display_str_list=(), use_normalized_coordinates=True):
    """Adds a bounding box to an image (numpy array)."""
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, thickness,
                                display_str_list, use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_mask_on_image_array(image, mask, color="red", alpha=0.4):
    """Draws mask on an image."""
    if image.dtype != np.uint8:
        raise ValueError("`image` not of type np.uint8")
    if mask.dtype != np.uint8:
        raise ValueError("`mask` not of type np.uint8")
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError("`mask` elements should be in [0, 1]")
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions don't match")

    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)
    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert("RGBA")
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert("L")
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert("RGB")))


def visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index,
                                               instance_masks=None, instance_boundaries=None,
                                               use_normalized_coordinates=False, max_boxes_to_draw=20,
                                               min_score_thresh=0.5, agnostic_mode=False,
                                               line_thickness=1, groundtruth_box_visualization_color="black",
                                               skip_scores=False, skip_labels=False, mask_alpha=0.4,
                                               plot_color=None):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_score_map = {}
    box_to_instance_boundaries_map = {}

    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ""
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in list(category_index.keys()):
                            class_name = category_index[classes[i]]["name"]
                        else:
                            class_name = "N/A"
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = "{}%".format(int(100 * scores[i]))
                    else:
                        float_score = ("%.2f" % scores[i]).lstrip("0")
                        display_str = "{}: {}".format(display_str, float_score)
                    box_to_score_map[box] = int(100 * scores[i])

                box_to_display_str_map[box].append(display_str)
                if plot_color is not None:
                    box_to_color_map[box] = plot_color
                elif agnostic_mode:
                    box_to_color_map[box] = "DarkOrange"
                else:
                    box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]

    if box_to_score_map:
        box_color_iter = sorted(box_to_color_map.items(), key=lambda kv: box_to_score_map[kv[0]])
    else:
        box_color_iter = box_to_color_map.items()

    for box, color in box_color_iter:
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            draw_mask_on_image_array(image, box_to_instance_masks_map[box], color=color, alpha=mask_alpha)
        if instance_boundaries is not None:
            draw_mask_on_image_array(image, box_to_instance_boundaries_map[box], color="red", alpha=1.0)
        draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, color=color,
                                          thickness=line_thickness,
                                          display_str_list=box_to_display_str_map[box],
                                          use_normalized_coordinates=use_normalized_coordinates)

    return image


def paste_instance_masks(masks, detected_boxes, image_height, image_width):
    """Paste instance masks to generate the image segmentation results."""
    def expand_boxes(boxes, scale):
        w_half = boxes[:, 2] * 0.5
        h_half = boxes[:, 3] * 0.5
        x_c = boxes[:, 0] + w_half
        y_c = boxes[:, 1] + h_half
        w_half *= scale
        h_half *= scale
        boxes_exp = np.zeros(boxes.shape)
        boxes_exp[:, 0] = x_c - w_half
        boxes_exp[:, 2] = x_c + w_half
        boxes_exp[:, 1] = y_c - h_half
        boxes_exp[:, 3] = y_c + h_half
        return boxes_exp

    _, mask_height, mask_width = masks.shape
    scale = max((mask_width + 2.0) / mask_width, (mask_height + 2.0) / mask_height)
    ref_boxes = expand_boxes(detected_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
    segms = []

    for mask_ind, mask in enumerate(masks):
        im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask[:, :]
        ref_box = ref_boxes[mask_ind, :]
        w = ref_box[2] - ref_box[0] + 1
        h = ref_box[3] - ref_box[1] + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)
        mask = cv2.resize(padded_mask, (w, h))
        mask = np.array(mask > 0.5, dtype=np.uint8)
        x_0 = min(max(ref_box[0], 0), image_width)
        x_1 = min(max(ref_box[2] + 1, 0), image_width)
        y_0 = min(max(ref_box[1], 0), image_height)
        y_1 = min(max(ref_box[3] + 1, 0), image_height)
        im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                            (x_0 - ref_box[0]):(x_1 - ref_box[0])]
        segms.append(im_mask)

    segms = np.array(segms)
    return segms


def plot_mask(color, alpha, original_image, mask):
    """Plot instance mask on image."""
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(original_image)
    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert("RGBA")
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert("L")
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    return np.array(pil_image.convert("RGB"))


def display_image(path_or_array, size=(10, 10)):
    """Display an image from path or array."""
    if isinstance(path_or_array, str):
        image = np.asarray(Image.open(open(path_or_array, "rb")).convert("RGB"))
    else:
        image = path_or_array
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def vild(image_path, category_name_string, params, plot_on=True, prompt_swaps=[]):
    """
    Run ViLD object detection on an image.

    Args:
        image_path: Path to the input image
        category_name_string: Semicolon-separated category names
        params: Tuple of (max_boxes, nms_thresh, min_rpn_score, min_box_area, max_box_area)
        plot_on: Whether to display visualization
        prompt_swaps: List of (old, new) string replacements for categories

    Returns:
        List of detected object names
    """
    # Preprocessing categories
    for a, b in prompt_swaps:
        category_name_string = category_name_string.replace(a, b)
    category_names = [x.strip() for x in category_name_string.split(";")]
    category_names = ["background"] + category_names
    categories = [{"name": item, "id": idx + 1} for idx, item in enumerate(category_names)]
    category_indices = {cat["id"]: cat for cat in categories}

    max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area = params
    fig_size_h = min(max(5, int(len(category_names) / 2.5)), 10)

    # Run ViLD model
    roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = session.run(
        ["RoiBoxes:0", "RoiScores:0", "2ndStageBoxes:0", "2ndStageScoresUnused:0",
         "BoxOutputs:0", "MaskOutputs:0", "VisualFeatOutputs:0", "ImageInfo:0"],
        feed_dict={"Placeholder:0": [image_path]})

    roi_boxes = np.squeeze(roi_boxes, axis=0)
    roi_scores = np.squeeze(roi_scores, axis=0)
    detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
    scores_unused = np.squeeze(scores_unused, axis=0)
    box_outputs = np.squeeze(box_outputs, axis=0)
    detection_masks = np.squeeze(detection_masks, axis=0)
    visual_features = np.squeeze(visual_features, axis=0)

    image_info = np.squeeze(image_info, axis=0)
    image_scale = np.tile(image_info[2:3, :], (1, 2))
    image_height = int(image_info[0, 0])
    image_width = int(image_info[0, 1])

    rescaled_detection_boxes = detection_boxes / image_scale

    # Read image
    image = np.asarray(Image.open(open(image_path, "rb")).convert("RGB"))
    assert image_height == image.shape[0]
    assert image_width == image.shape[1]

    # Filter boxes with NMS
    nmsed_indices = nms(detection_boxes, roi_scores, thresh=nms_threshold)
    box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * \
                (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

    valid_indices = np.where(
        np.logical_and(
            np.isin(np.arange(len(roi_scores), dtype=int), nmsed_indices),
            np.logical_and(
                np.logical_not(np.all(roi_boxes == 0., axis=-1)),
                np.logical_and(
                    roi_scores >= min_rpn_score_thresh,
                    np.logical_and(box_sizes > min_box_area, box_sizes < max_box_area)
                )
            )
        )
    )[0]

    detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
    detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
    detection_masks = detection_masks[valid_indices][:max_boxes_to_draw, ...]
    detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
    rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_draw, ...]

    # Compute text embeddings and scores
    text_features = build_text_embedding(categories)
    raw_scores = detection_visual_feat.dot(text_features.T)
    if FLAGS.use_softmax:
        scores_all = softmax(FLAGS.temperature * raw_scores, axis=-1)
    else:
        scores_all = raw_scores

    indices = np.argsort(-np.max(scores_all, axis=1))
    indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])

    # Get found objects
    found_objects = []
    for a, b in prompt_swaps:
        category_names = [name.replace(b, a) for name in category_names]

    for anno_idx in indices[0:int(rescaled_detection_boxes.shape[0])]:
        scores = scores_all[anno_idx]
        if np.argmax(scores) == 0:
            continue
        found_object = category_names[np.argmax(scores)]
        if found_object == "background":
            continue
        print("Found a", found_object, "with score:", np.max(scores))
        found_objects.append(category_names[np.argmax(scores)])

    if not plot_on:
        return found_objects

    # Visualization
    ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
    processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
    segmentations = paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

    if len(indices_fg) == 0:
        display_image(np.array(image), size=overall_fig_size)
        print("ViLD does not detect anything belonging to the given category")
    else:
        image_with_detections = visualize_boxes_and_labels_on_image_array(
            np.array(image),
            rescaled_detection_boxes[indices_fg],
            valid_indices[:max_boxes_to_draw][indices_fg],
            detection_roi_scores[indices_fg],
            numbered_category_indices,
            instance_masks=segmentations[indices_fg],
            use_normalized_coordinates=False,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_rpn_score_thresh,
            skip_scores=False,
            skip_labels=True)

        plt.imshow(image_with_detections)
        plt.title("ViLD detected objects and RPN scores.")
        plt.show()

    return found_objects


# Default category names for pick-and-place tasks
category_names = [
    'blue block', 'red block', 'green block', 'orange block', 'yellow block',
    'purple block', 'pink block', 'cyan block', 'brown block', 'gray block',
    'blue bowl', 'red bowl', 'green bowl', 'orange bowl', 'yellow bowl',
    'purple bowl', 'pink bowl', 'cyan bowl', 'brown bowl', 'gray bowl'
]

image_path = 'tmp.jpg'

# ViLD settings
category_name_string = ";".join(category_names)
max_boxes_to_draw = 8
prompt_swaps = [('block', 'cube')]
nms_threshold = 0.4
min_rpn_score_thresh = 0.4
min_box_area = 10
max_box_area = 3000
vild_params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area


if __name__ == "__main__":
    found_objects = vild(image_path, category_name_string, vild_params, plot_on=True, prompt_swaps=prompt_swaps)
    print("Found objects:", found_objects)