import cv2
import os
import numpy as np
def read_dic_face(file_list):
    data = np.loadtxt(file_list,dtype='str')
    dic_face = {}
    for i in range(len(data)):
        dic_face[int(data[i][0])] = data[i][1]
    
    return dic_face 
    
if __name__ == "__main__":    
    
    # 加载人脸字典
    dic_face = read_dic_face("face_list.txt")
    print(dic_face)
    
    # 加载Opencv人脸检测器
    faceCascade = cv2.CascadeClassifier('E:/1/PyCharm/python_class/practice_project_7/haarcascade_frontalface_alt.xml')

    # 加载训练好的人脸识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')


    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        
        # 读取一帧图像
        success, img = cap.read()
        
        if not success:
            continue
        
        # 转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 进行人脸检测
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(50, 50),flags=cv2.CASCADE_SCALE_IMAGE)
        
        # 遍历检测到的人脸
        for (x, y, w, h) in faces:
            # 画框
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
            # 进行人脸识别 
            id_face, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            print(confidence)
            # 检测可信度，这里是通过计算距离来计算可信度，confidence越小说明越近似 
            if (confidence < 100):
                str_face = dic_face[id_face]
                str_confidence = "  %.2f"%(confidence)
            else:
                str_face = "unknown"
                str_confidence = "  %.2f"%(confidence)
                
                
            # 检测结果文字输出
            cv2.putText(img, str_face+str_confidence, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        
        # 显示检测结果
        cv2.imshow("FACE",img)
        
        # 按键 "q" 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
          
    cap.release() 