import numpy as np
from PIL import Image
import cv2
import glob

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DeepLabModel(object):
	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = ['SemanticPredictions:0',"strided_slice:0"]
	INPUT_SIZE = 513

	def __init__(self, frozen_graph):
		self.graph = self.load_graph(frozen_graph)

		self.sess = tf.Session(graph=self.graph)

	def load_graph(self, frozen_graph):
		with tf.gfile.GFile(frozen_graph, "rb") as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())

		# We load the graph_def in the default graph
		with tf.Graph().as_default() as graph:
			seg_tensor = tf.import_graph_def(
				graph_def,
				input_map=None,
				return_elements=['SemanticPredictions:0'],
				name="",
				op_dict=None,
				producer_op_list=None
			)

		return graph

	def run(self, image):
		width, height = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
		target_size = (int(resize_ratio * width), int(resize_ratio * height))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		batch_seg_map,process_image = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})

		upsampled_output = tf.image.resize_nearest_neighbor(
			tf.expand_dims(batch_seg_map, axis=-1), [height, width])
		upsampled_seg = tf.squeeze(upsampled_output)
		upsampled_seg = tf.cast(upsampled_seg,tf.uint8)
		with tf.Session() as sess:
			upsampled_seg = sess.run(upsampled_seg)

		return resized_image, upsampled_seg


graph_pb = "./model.pb"
MODEL = DeepLabModel(graph_pb)

def run_visualization(DeepLabModel, path):
	image =cv2.imread(path)

	resized = cv2.resize(image, (513, 513))
	cv2.imwrite("temp.jpg", resized)
	orignal_im = Image.open("temp.jpg")
	resized_im, seg_map = DeepLabModel.run(orignal_im)

	image_RGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	seg_map = cv2.resize(seg_map, (image.shape[1], image.shape[0]))
	image = cv2.add(image_RGB, np.zeros(np.shape(image_RGB), dtype=np.uint8), mask=seg_map)
	seg = cv2.add(image_RGB, np.ones(np.shape(image_RGB), dtype=np.uint8) * 220, mask=seg_map)

	image = 220 - seg + image

	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	return image

def run(filepath):
	images = glob.glob(filepath)

	for image in images:
		result = run_visualization(MODEL, image)
		return result