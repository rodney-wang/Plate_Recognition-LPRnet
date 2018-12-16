import json, cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts, maxWidth=220, maxHeight=70):
    pts = np.array(pts).reshape([4, 2])
    #print(pts)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    dst = np.array([
        [0, 0],
       	[maxWidth - 1, 0],
       	[maxWidth - 1, maxHeight - 1],
       	[0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

if __name__ == '__main__':
    img = cv2.imread('./ch12_20180805095959.mp4_016125_box_1.jpg')
    j = json.load(open('./ch12_20180805095959.mp4_016125_box_1.jpg.out.json'))['text_lines'][0]
    points = np.array([[j['x0'], j['y0']], [j['x1'], j['y1']],
              [j['x2'], j['y2']], [j['x3'], j['y3']]])
    print(points)
    img = four_point_transform(img, points)
    cv2.imshow('a', img)
    cv2.waitKey()
