import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

#importa modelo do Yolo
model=YOLO("yolov8s.pt")

#Cria o primeiro retangulo para marcar a passagem

area1=[(312,388),(289,390),(474,469),(497,462)]

#Cria o segundo retangulo para marcar a passagem
area2=[(279,392),(250,397),(423,477),(454,469)]

#Funcao para saber as cordenadas do mouse
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

cap=cv2.VideoCapture("./Reconehcimento_Yolo/Tcc_Yolo/Contador_de_pessoas/peoplecount1.mp4")

#Arquivo com lista de possiveis classes que o Yolo detecta
my_file = open("./Reconehcimento_Yolo/Tcc_Yolo/Contador_de_pessoas/Classes.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker = Tracker()
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
#    frame=cv2.flip(frame,1)
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])

        #Arquivo com todas as classes
        c=class_list[d]

        #condição para detectar person
        if "person" in c:

            resutado = cv2.pointPolygonTest(np.array(area2, np.int32), ((x2, y2)), False)
            if resutado >= 0:
                #Cria o retangulo em volta da pessoa
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            
                #Circulo na quina do retanuglo 
                cv2.circle(frame, (x2, y2), 5, (255, 0, 255), -1)
                cv2.circle(frame, (x1, y2), 5, (0, 0, 255), -1)

                #Texto no comeco do retangulo
                cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
                cv2.putText(frame, "Pessoa detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    #Colocando cor na area 1
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str("1"),(504,471),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    #Colocando cor na area 2
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str("2"),(466,485),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(0)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

