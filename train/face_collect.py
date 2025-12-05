import cv2
import os

if __name__ == "__main__":
    str_face_id = ""
    index_photo=0

    # 加载训练好的人脸检测器
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        
        # 判断人脸id 是否为空，空的话创建face_id
        if str_face_id.strip()=="":
            str_face_id = input('Enter your face ID:')
            index_photo=0
            
            if not os.path.exists(str_face_id):
                os.makedirs(str_face_id)
          
        # 读取一帧图像
        success, img = cap.read()
        
        if not success:
            continue
        
        
        # 转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 进行人脸检测
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(50, 50),flags=cv2.CASCADE_SCALE_IMAGE)
        
        # 画框
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
        # 显示检测结果
        cv2.imshow("FACE",img)
        
        
        # 读取按键键值
        key =  cv2.waitKey(1) & 0xFF
          
        #  按键"c" 进行人脸采集
        if key == ord('c'):
            
            # 保存人脸
            for (x, y, w, h) in faces:
                roi = img[y:y+h,x:x+w]
                cv2.imwrite("%s/%d.jpg"%(str_face_id,index_photo),roi)
                index_photo = index_photo+1
            key = 0
        #  按键"x" 切换 人脸_id   
        elif key == ord('x'):
            str_face_id = ""
            key = 0
        # 按键 "q" 退出
        elif key ==  ord('q'):
            break
    cap.release() 