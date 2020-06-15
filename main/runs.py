import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util

from utils import visualization_utils as vis_util
import cv2
import time

cap = cv2.VideoCapture('/home/navan/mine/sack_counter/14.mp4')
# time.sleep(2.0)
(major, minor) = cv2.__version__.split(".")[:2]


count = 0
pos = 0
que = False
MODEL_NAME = '/home/navan/mine/sack_counter/out'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/navan/mine/sack_counter/object-detection.pbtxt'
NUM_CLASSES = 2


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)




PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            # fps = cap.get(CV_CAP_PROP_FPS,25)
            ret, image_np = cap.read()
            (H, W) = image_np.shape[:2]
            cv2.imwrite("filename.jpg", image_np)
            cv2.imwrite('o.jpg', cv2.resize(image_np, (H,W)))
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            min_score_thresh=.5

            print(len(boxes))
            for i in range(boxes.shape[0]):
                if scores is None or scores[i][i] > min_score_thresh:
                    box = boxes[i]
                    M = cv2.moments(box)
                    # print(box)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # print(M)

                    cv2.circle(image_np, (cX, cY), 7, (255, 255, 255), -1)
                    ymin = int(box[0][0]*H)
                    xmin = int(box[0][1]*W)
                    ymax = int(box[0][2]*H)
                    xmax = int(box[0][3]*W)
                    coord = (ymin, xmin, ymax, xmax)
                    initBB = (xmin, ymin, xmax, ymax)
                    # tracker.init(image_np, initBB)
                    # (success, box) = tracker.update(image_np)
                    # if success:
                    #     (x, y, w, h) = [int(v) for v in box]
                    #     cv2.rectangle(image_np, (x, y), (x + w, y + h),(0, 255, 0), 2)
                    centerCoord = (coord[0]+(coord[2]/2), coord[1]+(coord[3]/2))
                    cx = int((ymin+xmin))
                    cy = int((ymax+xmax))
                    # print("new",centerCoord[0],centerCoord[1])
                    # print("old",cx,cy)
                    # print((ymin,xmin))
                    pos = xmax
                    # print(ymax,xmax,H-10-1 // 1)
                    # cv2.circle(image_np, (int(centerCoord[0]),int(centerCoord[1])), radius=10, color=(0, 0, 255), thickness=-1)
                    # cv2.circle(image_np, (ymin,xmin), radius=10, color=(0, 0, 255), thickness=-1)
                    cv2.rectangle(image_np, (xmin,ymin), (xmax,ymax), (0, 0, 255), 30)
                    # cv2.circle(image_np, (int(ymin),int(ymax)), radius=10, color=(0, 255, 0), thickness=-1)
                    cv2.circle(image_np, (xmax,ymax), radius=10, color=(0, 255, 0), thickness=-1)
                    # print(cy)


                        #







            cv2.line(image_np, (0, H-100-1 // 1), (W, H-100-1 // 1), (255, 255, 0), 7)
            print(pos)
            # cv2.line(image_np, (0, H-10-1 // 1), (W, H-10-1 // 1), (255, 255, 0), 7)
            # cv2.line(image_np, (0, H-10-1 // 1), (W, H-10-1 // 1), (255, 255, 0), 7)

                # print((H-20-i // 1))
                # print("cy",pos)
            if pos == (H-100-1 // 1):
                count = count+1
                # print(pos,(H-100-7 // 1),(H-100-3 // 1))
                    # que== False
            cv2.putText(image_np, "COUNT =" + str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 255), 3, cv2.LINE_AA)

                # print(cy)
                #if cy == (H-20-i // 1):

            #
            # line1  = cv2.line(image_np, (0, H-400 // 1), (W, H-400 // 1), (255, 255, 0), 7)
            # line2  = cv2.line(image_np, (0, H-360 // 1), (W, H-360 // 1), (255, 255, 0), 7)
            # line3  = cv2.line(image_np, (0, H-320 // 1), (W, H-320 // 1), (255, 255, 0), 7)
            # line4  = cv2.line(image_np, (0, H-280 // 1), (W, H-280 // 1), (0, 255, 0), 7)
            # line5  = cv2.line(image_np, (0, H-240 // 1), (W, H-240 // 1), (0, 255, 0), 7)
            # line6  = cv2.line(image_np, (0, H-200 // 1), (W, H-200 // 1), (0, 255, 0), 7)
            # line7  = cv2.line(image_np, (0, H-180 // 1), (W, H-180 // 1), (255, 0, 0), 7)
            # line8  = cv2.line(image_np, (0, H-140 // 1), (W, H-140 // 1), (255, 0, 0), 7)
            # line9  = cv2.line(image_np, (0, H-100 // 1), (W, H-100 // 1), (255, 0, 0), 7)
            # line10 = cv2.line(image_np, (0, H-60 // 1), (W, H-60 // 1), (0, 0, 255), 7)
            # line11 = cv2.line(image_np, (0, H-20 // 1), (W, H-20 // 1), (0, 0, 255), 7)
            # print((0, H-400 // 1))
            # line12 = cv2.line(image_np, (0, H-360 // 1), (W, H-360 // 1), (255, 255, 0), 7)
            # line13 = cv2.line(image_np, (0, H-350 // 1), (W, H-350 // 1), (255, 255, 0), 7)
            # line14 = cv2.line(image_np, (0, H-340 // 1), (W, H-340 // 1), (255, 255, 0), 7)
            cv2.imshow('object detection', cv2.resize(image_np, (int(W/2),int(H/2))))
            # print(line1)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
