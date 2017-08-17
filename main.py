import os
import time
import argparse
import numpy as np
import tensorflow as tf

#Libraries for multiprocessing
from utils import FPS,WebcamVideoStream

#Libraries for image processing tasks
import cv2
import imutils

#Libraries from required for tensorflow labels
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#tensorboard log directory
LOGDIR = "tensorboard-log/"
CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np

def create_and_persist_session():
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)
            #writer = tf.summary.FileWriter("output", sess.graph)
            return sess

fps = FPS().start()
vs = WebcamVideoStream(src=0).start()
model_start = time.time()
session = create_and_persist_session()
model_loaded = time.time()
#print "Time taken to load the model: "+str(model_loaded-model_start)
 # loop over some frames...this time using the threaded stream
while fps._numFrames < args["num_frames"]:
# grab the frame from the threaded video stream and resize it
# to have a maximum width of 400 pixels
	frame = vs.read()
	#Format for the Mul:0 Tensor
	inputImage = cv2.resize(frame,dsize=(299,299), interpolation = cv2.INTER_CUBIC)
#	frame = imutils.resize(frame, width=400)
    outputImage = detect_objects(inputImage,session,detection_graph)
	if args["display"] > 0:
 		cv2.imshow('Input', frame)
		key = cv2.waitKey(1) & 0xFF

	print "The frame number is :"+str(fps._numFrames)
	merged_summary = tf.summary.merge_all()
	#print "The time required for classification is: "+str(processingTime)
	fps.update()

cv2.destroyAllWindows()
print "Average processing time is :"+str(totalTime/fps._numFrames)
fps.stop()
vs.stop()
