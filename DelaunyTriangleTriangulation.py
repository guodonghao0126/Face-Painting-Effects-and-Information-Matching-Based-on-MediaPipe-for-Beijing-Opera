import cv2
import numpy as np
import math

def constrainPoint(p, w, h):
  p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
  return p

def similarityTransform(inPoints, outPoints):
  s60 = math.sin(60*math.pi/180)
  c60 = math.cos(60*math.pi/180)
  inPts = np.copy(inPoints).tolist()
  outPts = np.copy(outPoints).tolist()
  #利用估计刚体变换，我们可以找到一个新的点，与已知的两个点共同构成一个等边三角形
  xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
  yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
  inPts.append([np.int(xin), np.int(yin)])
  xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
  yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
  outPts.append([np.int(xout), np.int(yout)])
  #利用estimateRigidTransform，我们可以进行相似性变换的计算
  tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
  return tform[0]

def rectContains(rect, point):
  if point[0] < rect[0]:
    return False
  elif point[1] < rect[1]:
    return False
  elif point[0] > rect[2]:
    return False
  elif point[1] > rect[3]:
    return False
  return True

def calculateDelaunayTriangles(rect, points):
  #构造一个二维细分曲面的实例
  subdiv = cv2.Subdiv2D(rect)
  #在细分曲面中插入点
  for p in points:
    subdiv.insert((int(p[0]), int(p[1])))
  #获取Delaunay三角形的测量值
  triangleList = subdiv.getTriangleList()
  #在点阵列中查找三角形的索引
  delaunayTri = []

  for t in triangleList:
    #函数`getTriangleList`返回的三角形信息包含三个点的六个坐标，分别为点1的x和y坐标，点2的x和y坐标，以及点3的x和y坐标
    #这些信息以'x1, y1, x2, y2, x3, y3'的格式呈现
    #我们将把每个三角形的这些信息存储为一个包含三个点的列表
    pt = []
    pt.append((t[0], t[1]))
    pt.append((t[2], t[3]))
    pt.append((t[4], t[5]))
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])
    if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
      #声明一个变量，该变量用于将三角形的信息存储为点列表中的索引
      ind = []
      #在点列表中，为每个顶点查找其索引
      for j in range(0, 3):
        for k in range(0, len(points)):
          if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
            ind.append(k)
        #将三角形的测量结果存储为索引列表
      if len(ind) == 3:
        delaunayTri.append((ind[0], ind[1], ind[2]))
  return delaunayTri

def applyAffineTransform(src, srcTri, dstTri, size):
  #已知两个三角形，我们需要找到一种仿射变换
  warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
  #将刚才计算出的仿射变换应用到src图像上
  dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
  return dst

def warpTriangle(img1, img2, t1, t2):
  #对于每个三角形，找到它的边界矩形
  r1 = cv2.boundingRect(np.float32([t1]))
  r2 = cv2.boundingRect(np.float32([t2]))
  #为每个矩形的左上角设置一个偏移点
  t1Rect = []
  t2Rect = []
  t2RectInt = []
  for i in range(0, 3):
    t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
    t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
  #通过填充三角形获得遮罩
  mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
  cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)
  #将图像应用于小矩形面片
  img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
  size = (r2[2], r2[3])
  img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
  img2Rect = img2Rect * mask
  #将矩形面片中的三角形区域复制到输出图像
  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect
