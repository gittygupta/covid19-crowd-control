from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import numpy as np
import cv2

model_path = 'models/mobilenet-v1-ssd-mp-0_675.pth'
label_path = 'models/voc-model-labels.txt'

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
'''
image = cv2.imread('image.jpg')
boxes, labels, probs = predictor.predict(image, 10, 0.4)

for i in range(boxes.size(0)):
    box = boxes[i, :]
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

    cv2.putText(image, label,
                (box[0]+20, box[1]+40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
cv2.imwrite('output.jpg', image)
'''

cap = cv2.VideoCapture('vids/vid2.mp4')

while(True):
    ret, image = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    
    cv2.imshow('frame',image)

cap.release()
cv2.destroyAllWindows()