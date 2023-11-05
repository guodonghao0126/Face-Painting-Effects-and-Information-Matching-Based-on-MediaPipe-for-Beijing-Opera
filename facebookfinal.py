import os
import tkinter
from tkinter import Label, Frame, Tk, Button,ttk,messagebox,Text
from tkinter.ttk import Combobox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import threading
import mediapipe as mp
import cv2
import math
import numpy as np
import DelaunyTriangleTriangulation as fbc
import csv
import speech_recognition as sr
import wave
import soundfile 
import pyaudio
import openpyxl
from PIL import Image,ImageDraw,ImageFont
from skimage.metrics import structural_similarity as ssim
VISUALIZE_FACE_POINTS = False

#定义特效图像和标注文件的路径
filters_config = {
    'facebook':
        [{'path': "filters/facebook.png",
          'anno_path': "filters/facebook_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'facebookex':
        [{'path': "filters/facebookex.png",
          'anno_path': "filters/facebook_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'facebook2':
        [{'path': "filters/facebook2.png",
          'anno_path': "filters/facebook2_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'facebook3':
        [{'path': "filters/facebook3.png",
          'anno_path': "filters/facebook_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
}
# 设置主题框架
root = Tk()
root.config(bg='GhostWhite')  # 设置背景色
root.title('京剧脸谱特效及信息匹配')
img_ori = None  # 设置用于显示图片的全局变量
#root.geometry("1000x400")
class GUI():
    def __init__(self):
        self.lb_text1 = Label(root, text="预览图片", font=('楷体', 14), bg='GhostWhite')
        self.lb_text1.place(x=90, y=20)
        
        # 设置显示子框架位置
        self.frm0 = Frame(root, width=480, height=360, bg="White")
        self.frm0.place(x=90, y=60)

        # 设置选择文件的按钮，绑定事件 self.chose_file
        self.button_chose1 = Button(root, text='选择图片', font=('楷体', 14), bg='White', activebackground='White',
                                    command=lambda: self.chose_file(),width=15)
        self.button_chose1.place(x=600, y=60)
        self.button_chose2 = Button(root, text='运行效果', font=('楷体', 14), bg='White', activebackground='White',
                                    command=lambda: runeffect(),width=15)
        self.button_chose2.place(x=600, y=240)
        self.button_chose1 = Button(root, text='确认', font=('楷体', 12), bg='White', activebackground='White',
                                    command=lambda: self.comget(),height=1)
        self.button_chose1.place(x=720, y=118)
        self.combobox = Combobox(root, values=["标准脸谱1", "标准脸谱2", "标准脸谱3"],width=10)
        self.combobox.place(x=600, y=120)
        #self.normal_ddl = Label(root, text='下拉框选项：')
        #self.ddl = ttk.Combobox(root)
        #self.ddl['value'] = ('标准脸谱1', '标准脸谱2', '标准脸谱3', '标准脸谱4')
        #self.ddl.place(x=600, y=100)
        self.button_chose3 = Button(root, text='匹配信息', font=('楷体', 14), bg='White', activebackground='White',
                                    command=lambda: self.showmessage(),width=15)
        self.button_chose3.place(x=600, y=180)
        
        self.file = None  # 用来记录每次上传的图片
        
    def chose_file(self, event=None):
        # 使用全局变量用来显示图片 img_ori
        global img_ori

        # 选择单个图片
        filename = askopenfilename(title='选择图片', filetypes=[(('JPG', '*.jpg')), ('All Files', '*')])
        ims = Image.open(filename)
        ims.save("filters/facebookex.png")
        self.file = ims.copy()  # 记录本次上传原始图片
        
        # 图片尺寸规格化
        w, h = ims.size
        if w > h:
            ime = ims.resize((480, int((480 * h / w))))
        else:
            ime = ims.resize((int(360 * w / h), 360))

        img_ori = ImageTk.PhotoImage(ime)
        lb1 = Label(self.frm0, image=img_ori, bg="white")  # 用来显示图片
        lb1.place(x=0, y=0)  # 设置图片的放置位置
        
        #检索图片相似度输出图片信息
    def showmessage(self, event=None):
        wb = openpyxl.load_workbook('facebook.xlsx')
        sheet = wb['京剧脸谱标注信息']
        path_= 'filters/facebookex.png'
        image1 = cv2.imread(path_)
        image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY) #  将图像转换为灰度图
        path = "facebookdataset"
        files= os.listdir(path) #得到文件夹下的所有文件名称
        for file in files: 
            image2 = cv2.imread(path+"/"+file)
            image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) #  将图像转换为灰度图
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            sim = ssim(image1, image2)
            if sim > 0.9:
                self.label_text1 = Label(root, text= sheet['B'+str(int(file.replace(".jpg",""))+1)].value, fg='black')
                #self.label_text1.place(x=900, y=60)
                #messagebox.showinfo("脸谱人物："+sheet['B'+str(int(file.replace(".jpg",""))+1)].value,sheet['R'+str(int(file.replace(".jpg",""))+1)].value )
                self.label_text2 = Label(root, text= sheet['R'+str(int(file.replace(".jpg",""))+1)].value, fg='black')
                #self.label_text2.place(x=900, y=120)
                message1 = tkinter.Message(root, bg="white", text="脸谱人物: "+sheet['B'+str(int(file.replace(".jpg",""))+1)].value, font="times 12 italic",width=300)
                message2 = tkinter.Message(root, bg="white", text="脸谱故事: "+sheet['R'+str(int(file.replace(".jpg",""))+1)].value, font="times 12 italic",width=300)
                message3 = tkinter.Message(root, bg="white", text="来源剧目: "+sheet['E'+str(int(file.replace(".jpg",""))+1)].value, font="times 12 italic",width=300)
                message4 = tkinter.Message(root, bg="white", text="剧目出处: "+sheet['G'+str(int(file.replace(".jpg",""))+1)].value, font="times 12 italic",width=300)
                message1.place(x=800, y=60)
                message3.place(x=800, y=110)
                message4.place(x=800, y=160)
                message2.place(x=800, y=220)
                #root.mainloop()
                #text1 = Text(root, width=40, height=20)
                #text1.pack()
                root.mainloop()
    
    #下拉框键
    def comget(self):
        print("method is called")
        print(self.combobox.get())
        num=self.combobox.get()
        global img_ori

        # 选择单个图片
        if num == "标准脸谱1":
            filename="filters/facebook.png"
        if num == "标准脸谱2":
            filename="filters/facebook2.png"
        if num == "标准脸谱3":
            filename="filters/facebook3.png"
        ims = Image.open(filename)
        ims.save("filters/facebookex.png")
        self.file = ims.copy()  # 记录本次上传原始图片
        
        # 图片尺寸规格化
        w, h = ims.size
        if w > h:
            ime = ims.resize((480, int((480 * h / w))))
        else:
            ime = ims.resize((int(360 * w / h), 360))

        img_ori = ImageTk.PhotoImage(ime)
        lb1 = Label(self.frm0, image=img_ori, bg="white")  # 用来显示图片
        lb1.place(x=0, y=0)  # 设置图片的放置位置

#运行特效，双线程
def runeffect():
    global cap
    global isFirstFrame
    global filters
    global multi_filter_runtime
    global count
    global filters_config
    global iter_filter_keys
    global sigma
    sh_img = cv2.imread(r"filters/facebookex.png", cv2.IMREAD_UNCHANGED)
    model_img = cv2.imread(r"filters/facebook.png", cv2.IMREAD_UNCHANGED)
    sh_img = cv2.resize(sh_img, (model_img.shape[1], model_img.shape[0]))
    cv2.imwrite("filters/facebookex.png", sh_img)
    sh_img = cv2.imread(r"filters/facebook3.png", cv2.IMREAD_UNCHANGED)
    model_img = cv2.imread(r"filters/facebook.png", cv2.IMREAD_UNCHANGED)
    sh_img = cv2.resize(sh_img, (model_img.shape[1], model_img.shape[0]))
    cv2.imwrite("filters/facebook3.png", sh_img)
    # process input from webcam or video file
    cap = cv2.VideoCapture(0)

    # Some variables
    count = 0
    isFirstFrame = True
    sigma = 50

    iter_filter_keys = iter(filters_config.keys())
    print("dd")
    print(next(iter_filter_keys))
    #filters, multi_filter_runtime = load_filter("facebookex")
    filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
    print("hh")
    print(filters, multi_filter_runtime)
    # The main loop

    thread_1 = threading.Thread(target=solve_image)
    thread_2 = threading.Thread(target=solve_speech)

    # start threads
    thread_1.start()
    thread_2.start()

    thread_1.join()
    thread_2.join()

    cap.release()
    cv2.destroyAllWindows()

#定义特效图像和标注文件的路径
def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]

    height, width = img.shape[:-1]

    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0

        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))

            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y

            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []

            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0


#定义特效图像和标注文件的路径
def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))

    return img, alpha

#定义特效图像和标注文件的路径
def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points

#定义特效图像和标注文件的路径
def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    """
    img:opecv格式
    cv2显示中文字符
    """
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "SimHei.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

#定义特效图像和标注文件的路径
def load_filter(filter_name="dog"):

    filters = filters_config[filter_name]

    multi_filter_runtime = []

    for filter in filters:
        temp_dict = {}

        img1, img1_alpha = load_filter_img(filter['path'], filter['has_alpha'])

        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha

        points = load_landmarks(filter['anno_path'])

        temp_dict['points'] = points

        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)
            print("mm")
            print(hull)
            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)
            #数分割的个数，返回三角形三点位置
            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue

        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap

        multi_filter_runtime.append(temp_dict)

    return filters, multi_filter_runtime

#图像部分
def solve_image():
    global isFirstFrame
    global filters
    global multi_filter_runtime
    global count
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:

            points2 = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # if face is partially detected
            if not points2 or (len(points2) != 75):
                continue

            ################ Optical Flow and Stabilization Code #####################
            img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if isFirstFrame:
                points2Prev = np.array(points2, np.float32)
                img2GrayPrev = np.copy(img2Gray)
                isFirstFrame = False

            lk_params = dict(winSize=(101, 101), maxLevel=15,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
            points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, points2Prev,
                                                            np.array(points2, np.float32),
                                                            **lk_params)

            # Final landmark points are a weighted average of detected landmarks and tracked landmarks

            for k in range(0, len(points2)):
                d = cv2.norm(np.array(points2[k]) - points2Next[k])
                alpha = math.exp(-d * d / sigma)
                points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
                #确保点在图像内，不在的变为边界点
                points2[k] = fbc.constrainPoint(points2[k], frame.shape[1], frame.shape[0])
                points2[k] = (int(points2[k][0]), int(points2[k][1]))

            # Update variables for next pass
            points2Prev = np.array(points2, np.float32)
            img2GrayPrev = img2Gray
            ################ End of Optical Flow and Stabilization Code ###############

            if VISUALIZE_FACE_POINTS:
                for idx, point in enumerate(points2):
                    cv2.circle(frame, point, 2, (255, 0, 0), -1)
                    cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
                cv2.imshow("landmarks", frame)

            for idx, filter in enumerate(filters):

                filter_runtime = multi_filter_runtime[idx]
                img1 = filter_runtime['img']

                points1 = filter_runtime['points']
                img1_alpha = filter_runtime['img_a']

                if filter['morph']:

                    hullIndex = filter_runtime['hullIndex']
                    dt = filter_runtime['dt']
                    hull1 = filter_runtime['hull']

                    # create copy of frame
                    warped_img = np.copy(frame)

                    # Find convex hull
                    hull2 = []
                    for i in range(0, len(hullIndex)):
                        hull2.append(points2[hullIndex[i][0]])

                    mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                    mask1 = cv2.merge((mask1, mask1, mask1))
                    img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                    # Warp the triangles
                    for i in range(0, len(dt)):
                        t1 = []
                        t2 = []

                        for j in range(0, 3):
                            t1.append(hull1[dt[i][j]])
                            t2.append(hull2[dt[i][j]])
                    #根据两组图像以及三角区域变形，图像融合
                        fbc.warpTriangle(img1, warped_img, t1, t2)
                        fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2
                else:
                    dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                    #仅仿射变换
                    tform = fbc.similarityTransform(list(points1.values()), dst_points)
                    # Apply similarity transform to input image
                    trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                    trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                    mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2

                frame = output = np.uint8(output)
                cv2.imshow("Face Filter",output)
            #cv2.putText(frame, "喊出“变脸转换脸谱！", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
            cv2AddChineseText(frame, "喊出“变脸转换脸谱！", (100, 200), textColor=(0, 255, 0), textSize=30)
            cv2.imshow("Face Filter", output)

            keypressed = cv2.waitKey(1) & 0xFF
    #        text = r.recognize_google(audio, language='zh-CN')
            if keypressed == 27:
                break
            # Put next filter if 'f' is pressed
            elif keypressed == ord('f'):
                try:
                    filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
                except:
                    iter_filter_keys = iter(filters_config.keys())
                    filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
            count += 1

#语音部分
def solve_speech():
    global isFirstFrame
    global filters
    global multi_filter_runtime
    global count
    global filters_config
    global iter_filter_keys
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "output.wav"

    while True:
        try:
            frames = []
            p = pyaudio.PyAudio()

            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

            print("recording")
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            print("done recording")

            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            r = sr.Recognizer()

            #test, samplerate = soundfile.read('output.wav')

            test = sr.AudioFile('output.wav')

            with test as source:
                audio = r.record(source)
            #  
            # type (audio)

            result=r.recognize_google(audio, language='zh-CN', show_all= True)
            print(result)
            if ("变" or "便") in result['alternative'][0]['transcript'] or ("变" or "便") in result['alternative'][1]['transcript'] or ("变" or "便") in result['alternative'][2]['transcript'] or ("变" or "便") in result['alternative'][3]['transcript']:
                print("cudsgiuewhuh")
                try:
                    filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
                except:
                    iter_filter_keys = iter(filters_config.keys())
                    filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
        except:
            continue

#运行
use = GUI() 
# root.wm_attributes("-topmost", 1)  # 窗口置顶
root.geometry('1180x520+{0}+{1}'.format(400, 120)) # 设置窗口大小和初始位置
# root.iconbitmap('test.ico') # 设置图标
root.mainloop()