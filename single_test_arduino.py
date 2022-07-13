import cv2  
import mediapipe as mp
import numpy as np
import random
import time
import playsound

#from Arduino import Arduino

#board = Aruduino('115200')

max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

rock_img = './data/rock_img.jpeg'
scissors_img = './data/scissors_img.jpeg'
paper_img  = './data/paper_img.jpeg'

lists = {0:rock_img, 1:scissors_img, 2:paper_img}
    
#for i in lists:
#    img2 = cv2.imread(lists[i], cv2.IMREAD_GRAYSCALE)
#    cv2.imshow('',img2)
#    time.sleep(1)
#    print(lists[i])

cap.set(3,1080)
cap.set(4,720)

list_rbt = [0,1,2]
robot_list = {0:'rock', 1:'scissors',2:'paper'}
rbt_time = 0

win_img = './data/win_img.jpeg'
lose_img = './data/lose_img.jpeg'
tie_img = './data/tie_img.jpeg'

win = 0
tie = 0
lose = 0

sound_file = './data/RSP_Voice.mp3'

rps_result = []

while cap.isOpened():

    ret, img = cap.read()
    if not ret:
        continue
    
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
            v = v2 - v1
            
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
            
            angle = np.degrees(angle)

            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            if idx in rps_gesture.keys():
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            if cv2.waitKey(1) & 0xFF == ord('y'):
                
#                time.sleep(0.5)
                playsound.playsound(sound_file, True)
#                time.sleep(0.5)

                # 해야 되는 것 : 
                
                # pyserial 사용법이나 c 코드를 python 코드로 바꿔주는 라이브러리 찾아야함 - pyserial 로 사용하면 됨. 즉 지금 가장 시급한 것은 arduino 를 어떻게 구성할 것인가? 이다. 

                # 시간이 된다면 module 로 만들어서 외부 프로그램에서 조금 더 
                # 쉽게 쓸 수 있도록 해보자

                idx_rbt = random.choice(list_rbt)
                rbt_time += 1

                if idx_rbt == 0: #which mean robot give us rock
                    
                    img2 = cv2.imread(lists[0], cv2.IMREAD_GRAYSCALE)
                    cv2.imshow('', img2)

                    #pyserial 해서 주먹 표현
                    print('rock')
                    
                    if idx == 0:
                        rps_result.append('tie')
                        tie += 1
                    elif idx == 5:
                        rps_result.append('win')
                        win += 1
                    elif idx == 9:
                        rps_result.append('lose')
                        lose += 1
                    else:
                        continue

                elif idx_rbt == 1: #which mean robot give us scissors

                    img3 = cv2.imread(lists[1], cv2.IMREAD_GRAYSCALE)
                    cv2.imshow('', img3)

                    #pyserial 해서 가위  표현
                    print('scissors')

                    if idx == 0:
                        rps_result.append('win')
                        win += 1
                    elif idx == 5:
                        rps_result.append('lose')
                        lose += 1
                    elif idx == 9:
                        rps_result.append('tie')
                        tie += 1
                    else:
                        continue

                elif idx_rbt == 2: #which mean robot give us paper

                    img4 = cv2.imread(lists[2], cv2.IMREAD_GRAYSCALE)
                    cv2.imshow('', img4)

                    #pyserial 해서 보자기 표현
                    print('paper')

                    if idx == 0:
                        rps_result.append('lose')
                        lose += 1
                    elif idx == 5:
                        rps_result.append('tie')
                        tie += 1
                    elif idx == 9:
                        rps_result.append('win')
                        win += 1
                    else:
                        continue

               # print(win)
               # print(tie)
               # print(lose)

                all_time = win + tie + lose
                print(len(rps_result))
                print(rps_result)
#                print(all_time)
#                if rps_result[0] == 'tie':
#                    img5 = cv2.imread(tie_img , cv2.IMREAD_GRAYSCALE)
#                    cv2.imshow('', img5)
#                elif rps_result[0] == 'lose':
#                    img6 = cv2.imread(lose_img , cv2.IMREAD_GRAYSCALE)
#                    cv2.imshow('', img6)
#                elif rps_result[0] == 'win':
#                    img7 = cv2.imread(win_img , cv2.IMREAD_GRAYSCALE)
#                    cv2.imshow('', img7)

                if all_time != 0:
                    win_rate = win * 100 / all_time
                else:
                    continue
        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    if len(rps_result) >=1:
        cv2.putText(img, text=rps_result[all_time-1], org = (300, 600), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255), thickness=2)
        
    else:
        pass
            
    cv2.imshow('Game', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

