from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime

class TLClassifier(object):
    def __init__(self, is_site):
        if is_site:
             PATH_TO_GRAPH = r'light_classification/model/real.pb'
        else:
            PATH_TO_GRAPH = r'light_classification/model/sim.pb'


        self.graph = tf.Graph()
        self.threshold = .5

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = np.asarray(image[:,:])
        np_image = np.asarray(image[:,:])
        image_expanded = np.expand_dims(np_image, axis=0)
        with self.graph.as_default():
            #with tf.Session(graph=self.graph) as sess:
            image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            detect_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            detect_scores = self.graph.get_tensor_by_name('detection_scores:0')
            detect_classes = self.graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num) = self.sess.run(
                [detect_boxes, detect_scores, detect_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
        
       
        if classes[0][0] == 1:
            # print('GREEN')
            return TrafficLight.GREEN
        elif classes[0][0] == 2:
            # print('RED')
            return TrafficLight.RED
        elif classes[0][0] == 3:
            # print('YELLOW')
            return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
