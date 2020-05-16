import numpy as np
import argparse
import time
import cv2
from scipy.spatial import distance as dist

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--pbtxt", type=str, 
	default='models/ssd_mobilenet_v1.pbtxt',
	help="path to tf 'deploy' pbtxt file")
ap.add_argument("--model", type=str, 
	default='models/ssd_mobilenet_v1.pb',
	help="path to tf mobile_net pre-trained model")
ap.add_argument("--input", type=str,
	default='videos/example_01.mp4',
	help="path to optional input video file")
ap.add_argument("--output", type=str,
	default='output/vid.avi',
	help="path to optional output video file")
ap.add_argument("--probability", type=float, 
	default=0.1,
	help="minimum probability to filter weak detections")
ap.add_argument("--threshold", type=float,
	default=20,
	help="max allowed distance between objects")
args = vars(ap.parse_args())

	
net = cv2.dnn.readNetFromTensorflow(args["model"], args["pbtxt"])

# if a video path was not supplied, grab a reference to the webcam
if args['input'] is None:
	print("[INFO] starting video stream...")
	vs = cv2.VideoCapture(0)
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# loop over frames from the video stream
while True:

	ret, frame = vs.read()
	if ret is False:
		break

	(H, W) = frame.shape[:2]

	rects = []
	centroids = []

	net.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
	preds = net.forward()	

	for detection in preds[0, 0]:
		score = float(detection[2])
		if score > args["probability"] and int(detection[1]) == 1:

			box = detection[3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			# draw a black rectangle around detected objects
			cv2.rectangle(frame, (startX, startY, endX, endY), (0, 0, 0), thickness=1)

			rects.append(box.astype("int"))

			centroids.append((int((startX + endX) / 2.0), int((startY + endY) / 2.0)))

	if len(rects) != 0:
		rects = np.array(rects)
		centroids = np.array(centroids)
	
		# distances between the centroids
		distance = dist.cdist(centroids, centroids)
		distance = np.array(distance)

		for i in range(distance.shape[0]):
			for j in range(distance.shape[1]):
				if i >= j:
					distance[i, j] = 9999

		# in order to perform this matching we must (1) find the
		# smallest value in each row and then (2) sort the row
		# indexes based on their minimum values so that the row
		# with the smallest value as at the *front* of the index
		# list
		rows = distance.min(axis=1).argsort()

		# next, we perform a similar process on the columns by
		# finding the smallest value in each column and then
		# sorting using the previously computed row index list
		cols = distance.argmin(axis=1)[rows]

		# minimum distance between two people for social distancing 
		threshold = 50

		# green boxes around objects closer than threshold
		for (row, col) in zip(rows, cols):
			if distance[row, col] < threshold:
				# getting the coordinates
				a1, b1, c1, d1 = rects[row, 0], rects[row, 1], rects[row, 2], rects[row, 3]
				a2, b2, c2, d2 = rects[col, 0], rects[col, 1], rects[col, 2], rects[col, 3]
				# adding green boxes
				cv2.rectangle(frame, (a1, b1, c1, d1), (0, 255, 0), thickness=2)
				cv2.rectangle(frame, (a2, b2, c2, d2), (0, 255, 0), thickness=2)
	

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
	
	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

if args['input'] is None:
	vs.stop()

# close any open windows
cv2.destroyAllWindows()
