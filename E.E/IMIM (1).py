import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import os ,datetime ,time
import 멀라 as car
import numpy as np ,pygame
import smtplib
from email.encoders import encode_base64
from email.header import Header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


x = datetime.datetime.now().strftime('%Y-%m-%d %H %M')

file = f'photo{x}.jpg'

# cap = []
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

img1 = car.cartoon   

detector = HandDetector(detectionCon=0.8)
 
 
class DragImg():
    def __init__(self, path, posOrigin, imgType):
 
        self.posOrigin = posOrigin
        self.imgType = imgType
        self.path = path
 
        if self.imgType == 'png':
            self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        else:
            self.img = cv2.imread(self.path)
 
        self.img = cv2.resize(self.img, (0,0),None,0.4,0.4)
 
        self.size = self.img.shape[:2]
 
    def update(self, cursor):
        ox, oy = self.posOrigin
        h, w = self.size
 
        # Check if in region
        if ox < cursor[0] < ox + w and oy < cursor[1] < oy + h:
            self.posOrigin = cursor[0] - w // 2, cursor[1] - h // 2
 
path = "ImagesPNG"   #이미지 불러오기
myList = os.listdir(path)
#print(myList)
 
listImg = []                       #이미지 리스트
for x, pathImg in enumerate(myList):
    if 'png' in pathImg:
        imgType = 'png'
    else:
        imgType = 'jpg'
    listImg.append(DragImg(f'{path}/{pathImg}', [25 + x * 100, 50], imgType))


pygame.init()
pygame.mixer.init()

sound = pygame.mixer.Sound('next.MP3') #사운드
sound.play()
prev = time.time()  

while True:
    success, img = cap.read()
    if not success : break
    img = cv2.flip(img, 1)

    hands,img = detector.findHands(img, flipType=False)

    
    cv2.rectangle(img, (0,0, 1280, 720), (255, 255, 255), -1) # 캠과 촬영한 이미지 합치기
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w1,h1,_ = img1.shape

    # print(w,h)

    img = img.copy()
    img[100:100+w1,300:300+h1,:] = img1




 
    if hands:
        lmList = hands[0]['lmList']
        # Check if clicked
        length, info,img= detector.findDistance(lmList[8], lmList[4],img)#손가락 포인트
        #print(length)
        if length < 40:           # 검지와 엄지의 거리가 40 이하 일때 이미지 움직이기
            cursor = lmList[8]
            for imgObject in listImg:
                imgObject.update(cursor)
 
    try:
 
        for imgObject in listImg:
 
            # Draw for JPG image
            h, w = imgObject.size
            ox, oy = imgObject.posOrigin
            if imgObject.imgType == "png":
                # Draw for PNG Images
                img = cvzone.overlayPNG(img, imgObject.img, [ox, oy])
                
            else:
                img[oy:oy + h, ox:ox + w] = imgObject.img
 
    except:
        pass


    if hands:
        lmList = hands[0]['lmList']  #검지와 약지로 사진 찍기
        # Check if clicked
        length, info = detector.findDistance(lmList[4], lmList[16])#손가락 포인트
        #print(length)
        if length < 25:
            cv2.imwrite(file, img)
            

            dst1 = img[100:580, 300:940].copy()

            cv2.imwrite('easy_on_the_eyes8.jpg', dst1)
            

            print(file,'저장됨')

            break
           
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows()


smtp = smtplib.SMTP('smtp.gmail.com', 587)
smtp.ehlo()      # say Hello
smtp.starttls()  # TLS 사용시 필요
smtp.login('아이디', '비밀번호')


msg = MIMEMultipart()


msg['To'] = input('전달받으실 이메일 입력 : ')
w

from email.header import Header
msg['Subject'] = Header(s='이지 온 디 아이즈 "대갈장군"이 왔습니다.', charset='utf-8')

from email.mime.text import MIMEText
body = MIMEText('이지 온 디 아이즈 "대갈장군"을 사용해 주셔서 감사합니다. team Easy on the eyes' , _charset='utf-8')
msg.attach(body)

files = list()
files.append('easy_on_the_eyes8.jpg')

from email.mime.base import MIMEBase
from email.encoders import encode_base64
for f in files:
    part = MIMEBase('application', "octet-stream")
    part.set_payload(open(f,"rb").read())
    encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(f))
    msg.attach(part)

smtp.sendmail('mrgoharm@gmail.com', msg['To'], msg.as_string())

smtp.quit()

print('발송했습니다.')













