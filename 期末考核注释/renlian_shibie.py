# 导入os模块：Python内置的文件/路径操作核心模块
# 核心功能：路径拼接、判断文件/目录是否存在、获取当前文件目录等
# 为什么写：跨平台处理路径（Windows用\，Linux用/），避免硬编码路径导致的兼容性问题
# 不写的影响：无法拼接face_data目录路径，只能写死绝对路径（如D:/code/face_data），代码移植性极差
import os

# 导入cv2模块：OpenCV（Open Source Computer Vision Library）核心库
# 核心功能：摄像头调用、图像读取/处理、人脸检测（Haar）、人脸识别（LBPH）、图像绘制（矩形/文字）
# 为什么写：是实现“摄像头实时人脸识别”的核心依赖，所有视觉相关操作都靠它
# 不写的影响：无法打开摄像头、无法检测/识别人脸、无法显示画面，整个程序核心功能失效
import cv2

# 导入numpy模块：Python数值计算基础库，主打高效的数组/矩阵操作
# 核心功能：此处用于快速读取face_list.txt文本文件，自动分割行/列
# 为什么写：比Python原生open()逐行读取更简洁高效，尤其适合结构化的文本（如ID 姓名）
# 不写的影响：需手动用open() + for循环读取文件，代码量增加，且分割字符串容易出错
import numpy as np

# 定义人脸资源根目录路径：拼接“当前文件所在目录”和“face_data”文件夹
# os.path.dirname(__file__)：获取当前.py文件的绝对目录（解决“运行目录不同导致相对路径失效”的问题）
# os.path.join()：跨平台路径拼接（自动适配Windows/Linux的路径分隔符）
# 为什么写：集中管理资源路径，后续修改目录只需改这一行，无需多处调整
# 不写的影响：后续每个资源路径都要重复拼接，代码冗余且易出错（比如漏写face_data目录）
FACE_DATA_DIR = os.path.join(os.path.dirname(__file__), "face_data")

# 拼接Haar人脸分类器文件路径：人脸检测的核心模型文件（XML格式，预训练的人脸特征库）
# 为什么写：明确指定分类器位置，让cv2能找到检测人脸的“模板”
# 不写的影响：cv2.CascadeClassifier会加载失败，报错“!empty()”，无法检测人脸
CASCADE_PATH = os.path.join(FACE_DATA_DIR, "haarcascade_frontalface_alt.xml")

# 拼接LBPH人脸识别模型文件路径：提前训练好的人脸特征库（yml格式，存储每个人的人脸特征）
# 为什么写：明确指定识别模型位置，让cv2能找到“人脸特征对照库”
# 不写的影响：recognizer.read()加载失败，无法识别人脸身份，只能检测人脸位置
TRAINER_PATH = os.path.join(FACE_DATA_DIR, "trainer.yml")

# 拼接人脸ID-姓名映射表路径：存储“数字ID→姓名”的文本文件（格式：1 张三 2 李四）
# 为什么写：将识别出的数字ID转换为人类可读的姓名，否则用户只能看到数字（无意义）
# 不写的影响：无法将ID转姓名，识别结果只能显示“1/2/3”，无法知道对应是谁
FACE_LIST_PATH = os.path.join(FACE_DATA_DIR, "face_list.txt")


def load_face_dictionary(list_path=FACE_LIST_PATH):
    """
    核心功能：加载ID-姓名映射表，返回“数字ID:姓名”的字典
    补充说明：移除类型注解后，参数默认值直接用FACE_LIST_PATH，无类型限制
    """
    # 判断映射表文件是否存在：提前拦截“文件丢失”的错误，避免后续读取崩溃
    # os.path.exists()：检查文件/目录是否存在，返回True/False
    # 为什么写：主动抛出“明确的错误提示”（告诉用户丢了哪个文件），而非让程序在读取时崩溃
    # 不写的影响：文件不存在时，np.loadtxt()会报错“File not found”，但错误信息不直观（用户不知道是映射表丢了）
    if not os.path.exists(list_path):
        # 主动抛出FileNotFoundError异常：自定义错误信息，明确指向丢失的文件路径
        # 为什么写：让用户能快速定位问题（比如face_list.txt没放在face_data目录）
        # 不写的影响：程序崩溃时只显示numpy的默认错误，用户难以排查
        raise FileNotFoundError(f"未找到人脸字典文件：{list_path}")

    # 读取映射表文本文件：np.loadtxt是numpy高效读取文本的方法
    # dtype="str"：强制将所有内容读取为字符串（避免ID被误读为浮点数，比如1变成1.0）
    # 为什么写：自动按空白符（空格/制表符）分割每行的ID和姓名，无需手动split()
    # 不写的影响：需手动写open(list_path) + for line in f: line.split()，代码繁琐且易出错（比如空行处理）
    data = np.loadtxt(list_path, dtype="str")

    # 初始化空字典：用于存储“ID:姓名”的映射关系
    # 为什么写：作为存储容器，逐行解析后存入字典，方便后续通过ID快速查姓名
    # 不写的影响：mapping未定义，后续mapping[int(row[0])]会报“name 'mapping' is not defined”
    mapping = {}

    # 遍历读取的每一行数据：逐行解析ID和姓名
    # 为什么写：将文件中的每行“ID 姓名”转换为字典的键值对
    # 不写的影响：无法解析文件内容，字典始终为空，后续ID转姓名失败
    for row in data:
        # 将每行第一列转为整数（ID），第二列作为姓名，存入字典
        # int(row[0])：文件中ID是字符串（如"1"），需转int才能和识别器返回的int型ID匹配
        # 为什么写：识别器predict返回的ID是int型（如1），如果字典键是str型（如"1"），会匹配失败
        # 不写的影响：face_dict.get(1)会返回None（因为键是"1"），所有识别结果都显示unknown
        mapping[int(row[0])] = row[1]

    # 返回映射字典：供后续识别函数调用，实现ID→姓名的转换
    # 为什么写：函数的核心输出，没有返回值则前面的解析操作无意义
    # 不写的影响：调用load_face_dictionary()后得不到任何数据，无法转姓名
    return mapping


def recognize_from_camera(duration_seconds=10, on_identity=None, silent=False):
    """
    核心功能：打开摄像头实时检测+识别人脸，限时/手动退出后返回识别结果
    补充说明：移除类型注解后，参数无类型限制，默认值直接赋值，返回值也无类型标注
    """
    # 检查Haar分类器文件是否存在：人脸检测的前提，无此文件无法找到人脸位置
    # 为什么写：主动拦截文件丢失错误，避免cv2加载分类器时崩溃
    # 不写的影响：cv2.CascadeClassifier(CASCADE_PATH)会加载失败，报错“!empty()”，无明确原因
    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(f"未找到 Haar 分类器：{CASCADE_PATH}")

    # 检查LBPH训练模型文件是否存在：人脸识别的前提，无此文件无法判断人脸身份
    # 为什么写：主动拦截文件丢失错误，避免cv2加载模型时崩溃
    # 不写的影响：recognizer.read(TRAINER_PATH)会失败，报错“!empty()”，无法识别人脸
    if not os.path.exists(TRAINER_PATH):
        raise FileNotFoundError(f"未找到训练模型：{TRAINER_PATH}")

    # 调用加载映射表函数：获取“ID:姓名”字典，为后续ID转姓名做准备
    # 为什么写：没有这一步，识别出的数字ID无法转换为姓名
    # 不写的影响：所有识别结果只能显示数字ID或unknown，无法显示姓名
    face_dict = load_face_dictionary()

    # 初始化Haar人脸检测器：加载分类器文件，创建检测实例
    # cv2.CascadeClassifier()：OpenCV内置的Haar特征检测器，专门用于目标检测（此处做人脸检测）
    # 为什么写：创建检测器实例，才能调用detectMultiScale()检测人脸
    # 不写的影响：无法检测人脸位置，后续识别逻辑无目标（不知道该识别哪里）
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # 初始化LBPH人脸识别器：创建识别实例（LBPH是OpenCV专为人脸识别设计的算法，抗光线/角度变化）
    # cv2.face.LBPHFaceRecognizer_create()：需安装opencv-contrib-python包（基础版opencv-python无此模块）
    # 为什么写：创建识别器实例，才能调用predict()识别人脸身份
    # 不写的影响：无法识别人脸身份，只能检测人脸位置（知道有脸，但不知道是谁）
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # 加载训练好的LBPH模型参数：将提前训练的人脸特征库载入识别器
    # 为什么写：识别器需要“对照特征库”才能判断当前人脸是谁
    # 不写的影响：识别器无特征数据，predict()返回随机ID/置信度，无法正确识别
    recognizer.read(TRAINER_PATH)

    # 打开默认摄像头：cv2.VideoCapture(0)表示系统默认摄像头（0=内置摄像头，1=外接摄像头）
    # 为什么写：获取摄像头的视频流，是“实时识别”的数据源
    # 不写的影响：无法获取视频帧，后续所有检测/识别逻辑无数据可处理
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开：避免摄像头被占用/无权限导致的无限空循环
    # cap.isOpened()：返回True（摄像头可用）/False（不可用）
    # 为什么写：提前拦截“摄像头打不开”的问题，避免程序进入死循环（cap.read()一直返回False）
    # 不写的影响：摄像头被占用时，程序会卡在while True里，一直打印“读取失败”（如果加了日志）或无响应
    if not cap.isOpened():
        # 主动抛出RuntimeError异常：明确告知用户摄像头无法打开
        # 为什么写：让用户排查摄像头权限/占用问题（比如被微信/Zoom占用）
        # 不写的影响：程序无响应，用户不知道是摄像头的问题
        raise RuntimeError("无法打开摄像头")

    # 初始化空集合：存储本次识别到的所有已知姓名（集合自动去重，避免重复记录同一人）
    # 为什么写：统计“本次识别到的所有人”，比如张三出现5次，集合只存1次，结果更准确
    # 不写的影响：无法统计所有识别到的人员，只能返回最后一次识别的姓名
    collected = set()

    # 初始化变量：记录最后一次识别的姓名（初始为None，表示还未识别到任何人）
    # 为什么写：保存最后一次识别结果，用于函数返回和回调函数
    # 不写的影响：无法返回“最后识别到的人”，函数返回的last_identity会报未定义
    last_identity = None

    # 记录开始时间：cv2.getTickCount()返回当前时钟周期数（高精度计时）
    # 为什么写：用于计算程序运行时长，实现“限时识别”（比如默认10秒后自动退出）
    # 不写的影响：无法判断运行时长，自动退出逻辑失效（只能手动按q退出）
    start = cv2.getTickCount()

    # 获取OpenCV时钟频率：将时钟周期数转换为秒（1秒 = cv2.getTickFrequency() 个周期）
    # 为什么写：把“时钟周期数”转成人类可读的“秒数”，便于判断是否达到duration_seconds
    # 不写的影响：无法将start的时钟数转为秒，elapsed计算错误，限时退出逻辑失效
    freq = cv2.getTickFrequency()

    # 非静默模式下打印提示信息：告知用户程序已启动及退出方式
    # silent=False：默认显示窗口+打印提示；silent=True：静默模式（无窗口/无提示）
    # 为什么写：提升用户体验，让用户知道程序在运行，以及如何手动退出
    # 不写的影响：用户不知道程序是否启动，也不知道按q可以退出
    if not silent:
        print("摄像头人脸识别已启动，按 'q' 退出窗口。")

    # 实时识别主循环：持续读取摄像头帧→检测人脸→识别身份，直到满足退出条件
    # 为什么写：实现“实时”识别，循环处理每一帧画面
    # 不写的影响：只处理一帧就结束，无法实现连续识别
    while True:
        # 读取摄像头一帧画面：cap.read()返回两个值（success=是否读取成功，frame=帧数据）
        # 为什么写：获取当前帧的图像数据，是检测/识别的基础
        # 不写的影响：无图像数据，后续所有操作都无法进行
        success, frame = cap.read()

        # 读取失败则跳过当前帧：避免因摄像头卡顿导致程序崩溃
        # 为什么写：摄像头偶尔卡顿会导致read()失败（success=False），frame为None，后续处理会报错
        # 不写的影响：frame为None时，cv2.cvtColor()会报错“(-215:Assertion failed) src.data”，程序崩溃
        if not success:
            continue

        # 将彩色帧转换为灰度图：人脸检测对灰度图更高效（减少通道数，计算量降为1/3）
        # cv2.cvtColor()：颜色空间转换，COLOR_BGR2GRAY=BGR（OpenCV默认）转灰度
        # 为什么写：Haar检测器对灰度图检测速度更快、准确率更高
        # 不写的影响：直接用彩色图检测，速度慢且易受颜色干扰，准确率下降
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 运行Haar人脸检测：返回人脸框坐标列表[(x,y,w,h), ...]
        # 参数说明：
        # - gray：灰度图（检测输入）
        # - scaleFactor=1.1：图像金字塔缩放系数（每次缩小10%，适配不同大小的人脸）
        # - minNeighbors=5：过滤噪声（需5个以上候选框重叠才认定为人脸，减少误检）
        # - minSize=(50,50)：最小人脸尺寸（过滤小于50x50的噪声，如小斑点）
        # - flags=cv2.CASCADE_SCALE_IMAGE：兼容旧版本OpenCV的检测标志
        # 为什么写：找到画面中所有人脸的位置，为后续识别提供目标区域
        # 不写的影响：无法知道人脸在哪，后续识别逻辑无目标
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # 遍历每个检测到的人脸框：对每一张人脸单独做识别
        # 为什么写：支持同时识别多个人脸（比如画面中有2个人，分别识别）
        # 不写的影响：只处理最后一个人脸框（或不处理），漏检多个人脸
        for (x, y, w, h) in faces:
            # 在原帧上绘制人脸框：可视化显示检测到的人脸位置
            # cv2.rectangle()参数：图像、左上角坐标、右下角坐标、颜色（BGR）、线宽
            # (255,0,0)=蓝色，线宽2=矩形边框厚度
            # 为什么写：让用户在窗口中直观看到“哪里检测到了人脸”
            # 不写的影响：用户看不到人脸位置，只能看到识别文字，体验差
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 识别人脸：对人脸区域（ROI）做LBPH识别，返回(ID, 置信度)
            # gray[y:y+h, x:x+w]：截取人脸区域（ROI，感兴趣区域），排除背景干扰
            # recognizer.predict()：核心识别方法，置信度越低→匹配度越高（0=完全匹配）
            # 为什么写：判断当前人脸对应的ID和匹配度
            # 不写的影响：只能检测人脸位置，无法知道是谁
            id_face, confidence = recognizer.predict(gray[y: y + h, x: x + w])

            # 判断置信度：<100视为识别成功（阈值可调整，越小越严格）
            # 置信度原理：计算当前人脸与特征库中人脸的差异值，值越小差异越小
            # 为什么写：区分“已知人脸”和“未知人脸”
            # 不写的影响：所有人脸都按已知处理，未知人脸会匹配错误ID（比如把陌生人识别为张三）
            if confidence < 100:
                # 根据ID查找姓名：未找到则返回"unknown"
                # face_dict.get()：避免KeyError（ID不在字典中时返回默认值）
                # 为什么写：容错处理（比如ID=999不在映射表中，不会崩溃）
                # 不写的影响：用face_dict[id_face]会报KeyError，程序崩溃
                identity = face_dict.get(id_face, "unknown")
            else:
                # 置信度过高（≥100），视为未知身份
                identity = "unknown"

            # 更新最后一次识别的姓名：覆盖为当前人脸的识别结果
            # 为什么写：保存最后一次识别到的姓名，用于函数返回和回调
            # 不写的影响：last_identity始终为None，无法返回最后识别结果
            last_identity = identity

            # 已知身份则加入集合：统计本次识别到的所有人员（自动去重）
            # 为什么写：记录“本次识别到的所有人”，比如张三、李四都出现过，集合存{张三, 李四}
            # 不写的影响：collected始终为空，无法返回“所有识别到的人员”
            if identity != "unknown":
                collected.add(identity)

            # 构造显示文本：姓名 + 置信度（保留2位小数），便于用户查看匹配度
            # 为什么写：让用户知道“识别出的是谁”+“匹配度有多高”
            # 不写的影响：窗口中只显示人脸框，看不到识别结果，体验差
            label = f"{identity} {confidence:.2f}"

            # 在人脸框上方绘制识别结果文本：可视化显示姓名和置信度
            # cv2.putText()参数：图像、文本、坐标、字体、字号、颜色（BGR）、线宽
            # (0,0,255)=红色，FONT_HERSHEY_SIMPLEX=常用无衬线字体，字号0.8，线宽2
            # 为什么写：将识别结果直观显示在画面上
            # 不写的影响：用户看不到识别结果，只能看到人脸框
            cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 计算已运行时长（秒）：(当前时钟数 - 开始时钟数) / 时钟频率
        # 为什么写：判断是否达到设定的duration_seconds（比如10秒），实现自动退出
        # 不写的影响：无法计算运行时长，自动退出逻辑失效
        elapsed = (cv2.getTickCount() - start) / freq

        # 非静默模式下显示窗口+检测退出按键
        if not silent:
            # 显示摄像头窗口：将带人脸框/识别文本的帧显示在窗口中
            # cv2.imshow()参数：窗口名称、显示的图像
            # 为什么写：让用户实时看到识别过程和结果
            # 不写的影响：用户看不到画面，只能靠打印信息判断，体验差
            cv2.imshow("Face Recognizer", frame)

            # 等待1ms并检测按键：按q则退出循环（必须加，否则窗口卡死）
            # cv2.waitKey(1)：等待1毫秒，返回按下的键值（&0xFF处理跨平台兼容）
            # ord("q")：获取q键的ASCII码（113）
            # 为什么写：实现“手动退出”功能，且避免窗口卡死（OpenCV窗口必须有waitKey才能刷新）
            # 不写的影响：窗口卡死无响应，无法手动退出，只能强制关闭程序
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 达到指定时长则自动退出循环：实现“限时识别”
        # duration_seconds=None：无限期运行（仅手动退出）；duration_seconds=10：10秒后退出
        # 为什么写：避免程序无限运行，适用于“打卡/签到”等限时场景
        # 不写的影响：程序只能手动按q退出，无法自动结束
        if duration_seconds is not None and elapsed >= duration_seconds:
            break

    # 释放摄像头资源：必须执行，否则摄像头会被该程序占用
    # 为什么写：释放后其他程序（如微信、Zoom）才能正常使用摄像头
    # 不写的影响：摄像头被占用，直到Python进程结束，其他程序无法使用
    cap.release()

    # 非静默模式下关闭所有OpenCV窗口：避免窗口残留（占用系统资源）
    # 为什么写：程序退出后清理窗口，避免桌面残留多个无效窗口
    # 不写的影响：窗口一直显示在屏幕上，需手动关闭
    if not silent:
        cv2.destroyAllWindows()

    # 如果传入回调函数，执行并传入最后一次识别的姓名
    # on_identity：可选的回调函数（比如识别后自动打卡、发送通知）
    # 为什么写：扩展程序功能，让识别结果可触发其他逻辑
    # 不写的影响：回调功能失效，无法实现“识别后自动处理”
    if on_identity:
        on_identity(last_identity)

    # 返回识别结果：最后一次识别的姓名 + 本次识别到的所有姓名集合
    # 为什么写：函数的核心输出，调用者可以获取识别结果（比如保存到数据库、打印）
    # 不写的影响：调用recognize_from_camera()后得不到任何结果，无法后续处理
    return last_identity, collected


# 程序入口：直接运行该文件时，执行摄像头人脸识别函数
# if __name__ == "__main__"：保证该代码块仅在“直接运行文件”时执行（导入时不执行）
# 为什么写：模块化设计，导入该模块时（如import xxx）不会自动启动摄像头
# 不写的影响：导入该模块时会立刻启动摄像头，不符合模块化规范
if __name__ == "__main__":
    # 调用核心函数：启动摄像头人脸识别（使用默认参数：10秒、非静默、无回调）
    # 为什么写：直接运行文件时触发识别逻辑，方便测试
    # 不写的影响：直接运行文件时无任何操作，程序启动即结束
    recognize_from_camera()