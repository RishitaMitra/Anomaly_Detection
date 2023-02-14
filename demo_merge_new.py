import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
#import os
import errno    #defines number of symbolic error codes
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')
confidenceThreshold = 0.5
NMSThreshold = 0.3

#modelConfiguration = 'yolo-coco/yolov3.cfg'
#modelWeights = 'yolo-coco/yolov3.weights'

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'cfg/yolov3.weights'
#modelConfiguration = 'yolov3.cfg'
#modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'
#labelsPath = 'yolo-coco/coco.names'
labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)



outputLayer = net.getLayerNames()
outputLayer = [outputLayer[i - 1] for i in net.getUnconnectedOutLayers()]  # [0]


def extractFrames(run, path):
	# Create output directory:
	try:
		os.mkdir(path)
	except OSError as exc:
		if exc.errno is not errno.EEXIST:
			raise
		pass

	video = cv2.VideoCapture(run)
	writer = None
	(W, H) = (None, None)

	try:
		prop = cv2.CAP_PROP_FRAME_COUNT
		total = int(video.get(prop))
		print("[INFO] {} total frames in video".format(total))
	except:
		print("Could not determine no. of frames in video")

	count = 0
	countern = 0
	f1 = open('videoplayback1.txt', 'a')
	while True:
		#countern=countern+1
		(ret, frame) = video.read()
		if not ret:
			break
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		# else:
			# if not countern%25==0:
				# continue
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		net.setInput(blob)
		layersOutputs = net.forward(outputLayer)

		boxes = []
		confidences = []
		classIDs = []
		i = 0
		print('A')
		for output in layersOutputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				if confidence > confidenceThreshold:
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype('int')
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# Apply Non Maxima Suppression
		detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
		if (len(detectionNMS) > 0):
			for i in detectionNMS.flatten():
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				print(labels[classIDs[i]])
				if labels[classIDs[i]] == 'person':
					print(labels[classIDs[i]])
					print('B')
					cnt = 0
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					if x > 0:  # if x becomes negative error will be generated
						#if h < 250:  # to get full body image
						print("image found")
						crop = frame[y:y + h, x:x + w]
						cv2.imwrite(os.path.join(path, "new{:d}.jpg".format(count)), crop)
						text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
						cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
						test_image =os.path.join(path, "new{:d}.jpg".format(count) )
						oriImg = cv2.imread(test_image)  # B,G,R order
						print('oriImg:',oriImg)
						candidate, subset = body_estimation(oriImg)
						print('candidate:',candidate)
						print('subset:',subset)
						#print('return:',bool(subset))
						# if(subset==[][]):
							# print('error')
						#cnt = 0
						canvas = copy.deepcopy(oriImg)
						#canvas = util.draw_bodypose(canvas, candidate, subset, cnt)
						try:
							canvas = util.draw_bodypose(canvas, candidate, subset, cnt)
						except:
							print("An exception occurred")
							f1.write('\n')
							f1.write('anomaly')
						# detect hand
						hands_list = util.handDetect(candidate, subset, oriImg)

						all_hand_peaks = []
						for x, y, w, is_left in hands_list:
						# cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
						# cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

							# if is_left:
							# plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
							# plt.show()
							peaks = hand_estimation(oriImg[y:y + w, x:x + w, :])
							peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
							peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
							# else:
							#    peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
							#     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
							#     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
							#     print(peaks)
							all_hand_peaks.append(peaks)

						canvas = util.draw_handpose(canvas, all_hand_peaks)

						plt.imshow(canvas[:, :, [2, 1, 0]])
						#plt.axis('off')
						#plt.show()
						print("pose estimated")
				else:
					print("no person")
					continue

					# if writer is None:
					#       fourcc = cv2.VideoWriter_fourcc(*'MJPG')
					#      writer = cv2.VideoWriter('chase_output.avi', fourcc, 30, (frame.shape[1], frame.shape[0]), True)
				# if writer is not None:
				#    writer.write(frame)
				#   print("Writing frame" , count+1)
				count = count + 1
				print("counter")

    #writer.release()
	video.release()

def main():
	# Call frame-extraction method with filename and directory parameters:
	extractFrames('videoplayback.mp4','playnew')


if __name__ == '__main__':
    main()