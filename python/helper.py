import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import cv2
import glob
import datetime

HW_trg=224

def process_image(input_img):
    input_img = tf.cast(input_img, tf.float32)
    # op_img = tf.keras.applications.mobilenet_v2.preprocess_input(
    # input_img, data_format=None)
    op_img = tf.keras.applications.vgg16.preprocess_input(
        input_img, data_format=None)
    return op_img


def unprocess_image(input_img):
    [0.485, 0.456, 0.406]
    b = tf.cast(input_img[:, :, 0] + tf.math.round(0.406 * 255), tf.int32)
    g = tf.cast(input_img[:, :, 1] + tf.math.round(0.456 * 255), tf.int32)
    r = tf.cast(input_img[:, :, 2] + tf.math.round(0.485 * 255), tf.int32)
    op_img = tf.stack([r, g, b], axis=2)
    return op_img
  
def scale_pad(ip_image, boxes, labels):

    ip_size = tf.cast(tf.shape(ip_image)[:-1], tf.float32)
    in_max = tf.math.argmax(ip_size)
    in_min = tf.math.argmin(ip_size)
    sf = HW_trg / ip_size[in_max]
    # Rescale the image (same ratio both sides)
    op_size = tf.cast(tf.math.round(sf * ip_size), tf.int32)
    ip_resize = tf.image.resize(ip_image, op_size, method=tf.image.ResizeMethod.BILINEAR)
    # Translate image within nominal HW_trg x HW_trg  window and add padding.
    # Translate shorter size image with padding operation.
    trans_ss_sf = tf.random.uniform(shape=[], minval=0.1, maxval=0.9)
    delta_ss = tf.cast(HW_trg - op_size[in_min], tf.float32)
    ss_offset_u = tf.cast(tf.math.floor(trans_ss_sf * delta_ss), tf.int32)
    ss_offset_l = HW_trg - (op_size[in_min] + ss_offset_u)
    # Translate longer size image with padding operation
    trans_ls_sf = tf.random.uniform(shape=[], minval=0.1, maxval=0.9)
    delta_ls = tf.cast(HW_trg - op_size[in_max], tf.float32)
    ls_offset_u = tf.cast(tf.math.floor(trans_ls_sf * delta_ls), tf.int32)
    ls_offset_l = HW_trg - (op_size[in_max] + ls_offset_u)
    if in_max == 1:
        paddings = [[ss_offset_u, ss_offset_l], [ls_offset_u, ls_offset_l], [0, 0]]
    else:
        paddings = [[ls_offset_u, ls_offset_l], [ss_offset_u, ss_offset_l], [0, 0]]
    ip_resize = tf.pad(ip_resize, paddings, "CONSTANT")
    # ip_resize = tf.image.rot90(ip_resize, k=1)
    # Rescale + offset all bounding box coordinates to match image transformations
    #
    ss_frac_offset = tf.cast(ss_offset_u / HW_trg, tf.float32)
    ls_frac_offset = tf.cast(ls_offset_u / HW_trg, tf.float32)
    ls_asp_ratio = tf.cast(1.0, tf.float32)
    ss_asp_ratio = tf.cast(ip_size[in_min] / ip_size[in_max], tf.float32)
    scale_boxes = _scale_boxes(boxes, ss_asp_ratio, ls_asp_ratio, ss_frac_offset, ls_frac_offset, in_max)

    # Output object centres (This is needed for yolo training)
    i_centre = [scale_boxes[..., 2] / 2 + scale_boxes[..., 0] / 2]
    j_centre = [scale_boxes[..., 3] / 2 + scale_boxes[..., 1] / 2]
    object_centres = tf.transpose(tf.concat((i_centre, j_centre), axis=0))

    return ip_resize, scale_boxes, object_centres, labels

# Obtain tensor slices from boxes scale add and conctenate portions (don't use mask (assignment) as in earlier versions)
def _scale_boxes(boxes, ss_asp_ratio, ls_asp_ratio, ss_frac_offset, ls_frac_offset, in_max):
    # Also correct for aspect ratio of smaller side and include offset
    if in_max == 1:
        r1 = (ss_asp_ratio) * boxes[..., 0] + ss_frac_offset
        r2 = (ls_asp_ratio) * boxes[..., 1] + ls_frac_offset
        r3 = (ss_asp_ratio) * boxes[..., 2] + ss_frac_offset
        r4 = (ls_asp_ratio) * boxes[..., 3] + ls_frac_offset
        scale_boxes = tf.transpose(tf.stack([r1, r2, r3, r4], axis=0))
    else:
        r1 = (ls_asp_ratio) * boxes[..., 0] + ls_frac_offset
        r2 = (ss_asp_ratio) * boxes[..., 1] + ss_frac_offset
        r3 = (ls_asp_ratio) * boxes[..., 2] + ls_frac_offset
        r4 = (ss_asp_ratio) * boxes[..., 3] + ss_frac_offset
        scale_boxes = tf.transpose(tf.stack([r1, r2, r3, r4], axis=0))
    return scale_boxes

def display_image(train_dataset):
    cntr = 0
    for img, boxes, obj_cen, labels in train_dataset.take(1):
        H_val = np.shape(img)[0]
        W_val = np.shape(img)[1]
        img = img.to_tensor(shape=[batch_size, HW_trg, HW_trg, 3])
        boxes = boxes[0]
        obj_cen = obj_cen[0]
        new_img = np.asarray(unprocess_image(img[0, :, :, :]))
        
        i = 0
        for b in boxes:
            print(b)
            print(labels[i])
            # get the bounding rects from dataset
            ymin = tf.cast((b[0] * HW_trg), tf.int32)
            xmin = tf.cast((b[1] * HW_trg), tf.int32)
            ymax = tf.cast((b[2] * HW_trg), tf.int32)
            xmax = tf.cast((b[3] * HW_trg), tf.int32)
            # print(xmin,xmax,ymin,ymax)
            # draw a green rectangle to visualize the bounding rect
            centre_int = (HW_trg * obj_cen)
            centre_int = tf.cast((HW_trg * obj_cen[i, :]), tf.int32)
            new_img = cv2.rectangle((new_img), (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            new_img = cv2.circle(new_img, (centre_int[1], centre_int[0]), 4, (255, 0, 0), 2)
            i += 1
        cntr = cntr + 1
        print(cntr)
    plt.imshow(new_img)
