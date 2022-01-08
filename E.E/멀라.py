import cv2,sys
import mediapipe as mp
import time, datetime, numpy as np
import imagee as im
from cvzone.HandTrackingModule import HandDetector
import cvzone, time,pygame


img1 = cv2.imread('images/caca.png',cv2.IMREAD_UNCHANGED)
img4 = cv2.imread('no.png',cv2.IMREAD_UNCHANGED)


TIMER = int(5) # 타이머 시간 (초)
x = datetime.datetime.now().strftime('%Y-%m-%d %H %M')

file = f'photo{x}.jpg'

cap = cv2.VideoCapture(0)
print(cap)

if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

cap.set(3, 700)
cap.set(4, 700)

detector = HandDetector(detectionCon=0.8)

class Button():
    def __init__(self,pos,text,size=(185,70)):  
        self.pos = pos
        self.size = size
        self.text = text
    
def btlist(keys=None):
    buttonList = []
    if keys is not None:
        for j in range(len(keys)):
            for i in range(len(keys[j])):
                buttonList.append(Button([300*i+70,80*j+50],keys[j][i]))
    return buttonList

def drawAll(img,buttonList):
    imgNew = np.zeros_like(img,np.uint8)
    for button in buttonList:
        x,y = button.pos
        w,h = button.size
        cvzone.cornerRect(imgNew,(x,y,w,h),20,rt=0)
        cv2.rectangle(imgNew,button.pos,(x+w,y+h),(0,255,0),cv2.FILLED)
        cv2.putText(imgNew,button.text,(x+15,y+60),
                cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),3)
    out = img.copy()
    alpha = 0.05
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img,alpha,imgNew,1-alpha,0)[mask]
    return out


keys = [['close','shot'],['cake','nono']]
   

buttonList = btlist(keys=keys)
print(buttonList)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    if not ret:
        break
    
    hands,img = detector.findHands(img, flipType=False)
    img = drawAll(img,buttonList)


    if hands : 
        hand = hands[0]
        lmList = hands[0]['lmList']

        length, info,img = detector.findDistance(lmList[4], lmList[8],img)#손가락 포인트

        for button in buttonList:
            x,y = button.pos
            w,h = button.size

            if (x<lmList[4][0]<x+w) and (y<lmList[4][1]<y+h) :
                if hand['type'] in ['Left','Right']:                
                    cv2.rectangle(img,(x,y),(x+w,y+h),(170,0,170),cv2.FILLED)
                    cv2.putText(img,button.text,(x+15,y+60),
                            cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),3)
                    l, _ = detector.findDistance(lmList[4],lmList[12])

                    if length <25 :
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),cv2.FILLED)
                        cv2.putText(img,button.text,(x+15,y+60),
                            cv2.FONT_HERSHEY_PLAIN,5,(255,255,255),5)

                        time.sleep(0.15)
                        if button.text =='close':
                            cap.release()
                            cv2.destroyAllWindows()

                        time.sleep(0.15)
                        if button.text =='nono':
                            while True:
                                ret, frame = cap.read()
                                    
                                frame = cv2.flip(frame, 1)
                                hands,frame = detector.findHands(frame, flipType=False)


                                w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                # w1,h1,_ = img1.shape
                                # print(img1.shape)    
                                mask = img4[:, :, 3]
                                img3 = img4[:,:,:-1]
                                h1, w1 = mask.shape[:2]
                                crop = frame[10:10+w1,70:70+h1]
                                

                                cv2.copyTo(img3,mask,crop)
                                

                                
                                if hands:
                                    lmList = hands[0]['lmList']

                                    length, info,frame = detector.findDistance(lmList[4], lmList[16],frame)#손가락 포인트

                                    if length < 30: # 검지와 약지의 거리가 40 이하일때 촬영

                                        pygame.init()
                                        pygame.mixer.init()
                                
                                        eat_sound = pygame.mixer.Sound('54321(1).mp3') #사운드
                                        eat_sound.play()

                                        prev = time.time()  

                                        while TIMER >= 0: 
                                            
                                                # 타이머 동작
                                            ret, frame = cap.read()
                                            frame = cv2.flip(frame, 1)
                                            
                                            w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                            h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                            # w1,h1,_ = img1.shape
                                            # print(img1.shape)    
                                            mask = img4[:, :, 3]
                                            img3 = img4[:,:,:-1]
                                            h1, w1 = mask.shape[:2]
                                            crop = frame[10:10+w1,70:70+h1]
                                            

                                            cv2.copyTo(img3,mask,crop)

                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            cv2.putText(frame, str(TIMER),
                                                        (200, 250), font,
                                                        7, (255, 0, 255),
                                                        4, cv2.LINE_AA)
                                            
                                            cv2.imshow('img', frame)
                                            cv2.waitKey(1)

                                            # current time
                                            cur = time.time()
                            
                                    
                                            if cur-prev >= 1:
                                                prev = cur
                                                TIMER = TIMER-1

                                        else:
                                            ret, frame = cap.read()
                                            frame = cv2.flip(frame, 1)
                                            w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                            h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                            # w1,h1,_ = img1.shape
                                            # print(img1.shape)    
                                            mask = img4[:, :, 3]
                                            img3 = img4[:,:,:-1]
                                            h1, w1 = mask.shape[:2]
                                            crop = frame[10:10+w1,70:70+h1]
                                            

                                            cv2.copyTo(img3,mask,crop)
                                            im.line_size = 5  # 선의 두께         
                                            im.blur_value = 5
                                            edges = im.edge_mask(frame, im.line_size, im.blur_value)
                                                                    
                                            total_color = 15 #색상의 수 
                                            img1 = im.color_quantization(frame, total_color)
                                            blurred = cv2.bilateralFilter(img1, d=9, sigmaColor=20,sigmaSpace=20) #양뱡향 필터
                                            cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)


                                            cv2.imshow('img', frame)
                                            
                                
                                            cv2.waitKey(2000)
                                
                                            cv2.imwrite(file, cartoon)    #타이머 종료 후 이미지 저장
                                            print(file,'저장됨')
                                        
                                        break

                                        cv2.destroyAllWindows()
                                cv2.imshow('img', frame)
                                cv2.waitKey(1)
                            cap.release()
                            cv2.destroyAllWindows()


                        time.sleep(0.15)
                        if button.text =='cake' : 
                            while True:
                                ret, frame = cap.read()
                                    
                                frame = cv2.flip(frame, 1)
                                hands,frame = detector.findHands(frame, flipType=False)


                                w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                # w1,h1,_ = img1.shape
                                # print(img1.shape)    
                                mask = img1[:, :, 3]
                                img2 = img1[:,:,:-1]
                                h1, w1 = mask.shape[:2]
                                crop = frame[10:10+w1,70:70+h1]
                                

                                cv2.copyTo(img2,mask,crop)
                                

                                
                                if hands:
                                    lmList = hands[0]['lmList']

                                    length, info,frame = detector.findDistance(lmList[4], lmList[16],frame)#손가락 포인트

                                    if length < 30: # 검지와 약지의 거리가 40 이하일때 촬영

                                        pygame.init()
                                        pygame.mixer.init()
                                
                                        eat_sound = pygame.mixer.Sound('54321(1).mp3') #사운드
                                        eat_sound.play()

                                        prev = time.time()  

                                        while TIMER >= 0: 
                                            
                                                # 타이머 동작
                                            ret, frame = cap.read()
                                            frame = cv2.flip(frame, 1)
                                            
                                            w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                            h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                                            mask = img1[:, :, 3]
                                            img2 = img1[:,:,:-1]
                                            h1, w1 = mask.shape[:2]
                                            crop = frame[10:10+w1,70:70+h1]
                                            

                                            cv2.copyTo(img2,mask,crop)

                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            cv2.putText(frame, str(TIMER),
                                                        (200, 250), font,
                                                        7, (255, 0, 255),
                                                        4, cv2.LINE_AA)
                                            
                                            cv2.imshow('img', frame)
                                            cv2.waitKey(1)

                                            # current time
                                            cur = time.time()
                            
                                    
                                            if cur-prev >= 1:
                                                prev = cur
                                                TIMER = TIMER-1

                                        else:
                                            ret, frame = cap.read()
                                            frame = cv2.flip(frame, 1)
                                            w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                            h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                                            mask = img1[:, :, 3]
                                            img2 = img1[:,:,:-1]
                                            h1, w1 = mask.shape[:2]
                                            crop = frame[10:10+w1,70:70+h1]
                                            

                                            cv2.copyTo(img2,mask,crop)

                                            im.line_size = 5  # 선의 두께         
                                            im.blur_value = 5
                                            edges = im.edge_mask(frame, im.line_size, im.blur_value)
                                                                    
                                            total_color = 15 #색상의 수 
                                            img1 = im.color_quantization(frame, total_color)
                                            blurred = cv2.bilateralFilter(img1, d=9, sigmaColor=20,sigmaSpace=20) #양뱡향 필터
                                            cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)


                                            cv2.imshow('img', frame)
                                            
                                
                                            cv2.waitKey(2000)
                                
                                            cv2.imwrite(file, cartoon)    #타이머 종료 후 이미지 저장
                                            print(file,'저장됨')
                                        
                                        break

                                        cv2.destroyAllWindows()
                                cv2.imshow('img',frame)

                                cv2.waitKey(1)
                            cap.release()
                            cv2.destroyAllWindows()




                        time.sleep(0.15)
                        if button.text =='shot':
                            while True:
                                ret1, img = cap.read()
                                    
                                img = cv2.flip(img, 1)
                                hands,img = detector.findHands(img, flipType=False)
                              
                                if hands:
                                    lmList = hands[0]['lmList']

                                    length, info,img = detector.findDistance(lmList[4], lmList[16],img)#손가락 포인트

                                    if length < 30: # 검지와 약지의 거리가 40 이하일때 촬영

                                        pygame.init()
                                        pygame.mixer.init()
                                
                                        eat_sound = pygame.mixer.Sound('54321(1).mp3') #사운드
                                        eat_sound.play()
                                        prev = time.time()  
                                        
                                      

                                        







                                        while TIMER >= 0:       # 타이머 동작
                                            ret1, img = cap.read()
                                            img = cv2.flip(img, 1)

                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            cv2.putText(img, str(TIMER),
                                                        (200, 250), font,
                                                        7, (255, 0, 255),
                                                        4, cv2.LINE_AA)
                                            
                                            cv2.imshow('img', img)
                                            cv2.waitKey(1)

                                            # current time
                                            cur = time.time()
                            
                                    
                                            if cur-prev >= 1:
                                                prev = cur
                                                TIMER = TIMER-1

                                        else:
                                            ret1, img = cap.read()
                                            img = cv2.flip(img, 1)
                                            cv2.imshow('img', img)
                                            pygame.init()
                                            pygame.mixer.init()

                                            sound = pygame.mixer.Sound('bom.wav') #사운드
                                            sound.play()

                                
                                            cv2.waitKey(2000)
                                            im.line_size = 5  # 선의 두께              # 이미지 캐릭터화 
                                            im.blur_value = 5
                                            edges = im.edge_mask(img, im.line_size, im.blur_value)
                                                                    
                                            total_color = 15 #색상의 수 
                                            img1 = im.color_quantization(img, total_color)
                                            blurred = cv2.bilateralFilter(img1, d=9, sigmaColor=20,sigmaSpace=20) #양뱡향 필터
                                            cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)


                                            cv2.imwrite(file, cartoon)
                                            
                                
                                            # cv2.imwrite(file, img)    #타이머 종료 후 이미지 저장
                                            print(file,'저장됨')
                                        
                                        break

                                        cv2.destroyAllWindows()
                                cv2.imshow('img',img)

                                cv2.waitKey(1)
                            cap.release()
                            cv2.destroyAllWindows()


     

    cv2.imshow('img', img)


    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

