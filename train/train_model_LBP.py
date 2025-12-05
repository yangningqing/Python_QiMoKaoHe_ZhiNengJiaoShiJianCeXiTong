import cv2
import os
import numpy as np

# 获取所有文件（人脸id）
def get_face_list(path):
    for root,dirs,files in os.walk(path):
        if root == path:
            return dirs
        

if __name__ == "__main__":
    
    # 创建人脸识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # 用来存放人脸id的字典
    # 构建人脸编号 和 人脸id 的关系
    dic_face = {}

    # 人脸存储路径

    base_path = "E:/1/PyCharm/python_class/practice_project_7"
    
    # 获取人脸id
    face_ids = get_face_list(base_path)
    print(face_ids)
    # 用来存放人脸数据与id号的列表
    faceSamples=[]
    ids = []
    
    # 遍历人脸id命名的文件夹
    for i, face_id in enumerate(face_ids):
        
        # 人脸字典更新
        dic_face[i] = face_id
            
        # 获取人脸图片存放路径
        path_img_face = os.path.join(base_path,face_id)
        
        for face_img in os.listdir(path_img_face):
            # 读取以.jpg为后缀的文件
            if face_img.endswith(".jpg"):
                file_face_img = os.path.join(path_img_face,face_img)
                
                # 读取图像并转换为灰度图
                img = cv2.imread(file_face_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 保存图像和人脸ID
                faceSamples.append(img)
                ids.append(i)
    
    print(dic_face)
    
    # 进行模型训练    
    recognizer.train(faceSamples, np.array(ids))

    # 模型保存 
    recognizer.save('trainer.yml')                
    
    # 进行字典保存
    with open("face_list.txt",'w') as f:
        for face_id in dic_face:
            f.write("%d %s\n"%(face_id,dic_face[face_id]))
    
        
        
        
        


                
      