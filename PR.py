# coding=utf-8
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
import matplotlib.pyplot as plt

CLASS = ''

def ROI(xMin1, yMin1, xMax1, yMax1, xMin2, yMin2, xMax2, yMax2):
    '''
    计算ROI，面积比大于阈值，返回1
    否则返回0
    '''
    # 矩阵1和2的面积
    area1 = (xMax1 - xMin1) * (yMax1 - yMin1)
    area2 = (xMax2 - xMin2) * (yMax2 - yMin2)
    print(area1, area2)

    # 相交矩阵的坐标，面积
    xMin3 = max(xMin1, xMin2)
    yMin3 = max(yMin1, yMin2)
    xMax3 = min(xMax1, xMax2)
    yMax3 = min(yMax1, yMax2)
    # 当没有交集，把面积置0
    if (xMin3 > xMax3) or (yMin3 > yMax3):
        area3 = 0
    else:
        area3 = (xMax3 - xMin3) * (yMax3 - yMin3)
    print(area3, '\n')

    # 并集面积
    area4 = area1 + area2 - area3

    # ROI, 阈值为threshold
    if float(area3) / area4 >= threshold:
        return 1
    else:
        return 0


def read_data(annotationpath, labelspath, classes, true, scores, threshold):
    '''
    读取数据，并将结果加入scores，true
    '''
    res = open(annotationpath, 'r')
    for eachline in res:
        info = eachline.split()
        labelname = info[0] + '.txt'
        score = info[1]
        # xMin1, yMin1, xMax1, yMax1 = float(info[2]), float(info[3]), float(info[4]), float(info[5])
        if info[2] != CLASS:
            continue

        xMin1, yMin1, xMax1, yMax1 = float(info[3]), float(info[4]), float(info[5]), float(info[6])
        f = open(labelspath + labelname)
        flag = 0
        for eachline in f:
            line = eachline.split()
            if len(line) == 1:
                continue
            if line[4] != classes:
                continue
            xMin2, yMin2, xMax2, yMax2 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
            flag = ROI(xMin1, yMin1, xMax1, yMax1, xMin2, yMin2, xMax2, yMax2)
            if flag == 1:
                break
        # 得分小于0.3的，抛弃
        # if float(score) <= 0.3:
        #     continue
        scores.append(float(score))
        true.append(flag)


def drawpicture(recall, precision, AP, classes):
    '''
    绘制结果图片
    '''
    plt.figure(1)  # 创建图表1
    plt.title('Precision/Recall Curve {} mAP={:.3}'.format(classes, AP))  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')

    plt.figure(1)
    plt.plot(recall, precision, color='blue')

    plt.savefig('{}.png'.format(classes))
    plt.show()


def getAP(precision, recall):
    '''
    用来计算AP值，但是我这个算法太简单粗暴，和sklearn算出来的结果误差蛮大的。
    暂时弃用
    '''
    AP = 0
    for i in range(len(recall)):
        AP = AP + precision[i] * (1.0 / len(recall))
    print(AP)
    return AP


if __name__ == '__main__':
    classes = 'gangjin'
    CLASS = classes
    threshold = 0.5
    # annotationpath = r'data\results\comp4_782f0b56-4218-4fd2-bfb8-60a39602e790_det_test_{}.txt'.format(classes)
    annotationpath = r'data/VOCdevkit2007/results/VOC2007/Main/result1.txt'
    labelspath = r'data/VOCdevkit2007/VOC2007/Labels/'
    scores = []
    true = []
    read_data(annotationpath, labelspath, classes, true, scores, threshold)
    print(scores)
    print(true)
    precision, recall, thresholds = precision_recall_curve(true, scores)
    print(precision)
    print(recall)
    # print thresholds
    precision[0] = 0
    # AP = getAP(precision, recall)
    AP = average_precision_score(true, scores)
    drawpicture(recall, precision, AP, classes)
