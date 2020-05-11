from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import numpy as np
import cv2

cyan = (255, 255, 0)
white = (255, 255, 255)
red = (0, 0, 255)
width = 2
area_ratio = 0.75
distance_factor = 1

def change_brightness(image, value):
    return cv2.add(image, np.array([float(value)]))

model_path = 'models/mobilenet-v1-ssd-mp-0_675.pth'
label_path = 'models/voc-model-labels.txt'

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

cap = cv2.VideoCapture('vids/vid.mp4')

while(True):
    ret, image = cap.read()
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    
    # logic
    # only people boxes
    people = []
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        if label.split(':')[0] != 'person':
            continue
        people.append(box)
    print(image.shape)

    # change brightness
    low_bright = change_brightness(image, -50)
    
    # restoring brightness of boxes
    for box in people:
        low_bright[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :] = change_brightness(image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :], 30)
    image = low_bright
    
    # check
    colors = [white for _ in range(len(people))]
    for i in range(len(people) - 1):
        color = white
        box_i = people[i]
        for j in range(i, len(people)):
            box_j = people[j]
            area_i = int(box_i[2] - box_i[0]) * int(box_i[3] - box_i[1])
            area_j = int(box_j[2] - box_j[0]) * int(box_j[3] - box_j[1])
            if area_j >= area_i * area_ratio and area_j <= area_i / area_ratio:
                # distance between right shoulder of left person and left shoulder of right person
                social_distance = distance_factor * max(int(box_i[2] - box_i[0]), int(box_j[2] - box_j[0]))   # currently social_distance = 'Wider shoulder among the 2'
                if abs(int(box_i[2] - box_j[0])) < social_distance or abs(int(box_i[0] - box_j[2])) < social_distance:
                    colors[i] = red
                    colors[j] = red        
        
    # coloring
    for i in range(len(people)):
        box = people[i]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors[i], width)
        
    
    cv2.imshow('frame',image)

cap.release()
cv2.destroyAllWindows()

