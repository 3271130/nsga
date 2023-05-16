import numpy as np
import xlrd
import math

def excel2m(path,num):#读excel数据转为矩阵函数
    data = xlrd.open_workbook(path)
    table = data.sheets()[num] # 获取excel中第num个sheet表
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows, ncols))
    for x in range(ncols):
        cols = table.col_values(x)
        cols1 = np.matrix(cols)  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x] = cols1 # 把数据进行存储
    return datamatrix

def searchCut(X1,Nst,Na,file):
    timeRobot=excel2m(file,2)[3] # 机器人的平均时间
    sum = np.sum(timeRobot) # 完成所有任务的时间
    sum_left = 0
    Cut = np.zeros(Nst) # 用来存储当前染色体的割点坐标

    for i in range(len(Cut)):
        Cut[i] = Na
    for i in range(int(math.log2(Nst))): # 工作站要求是2的幂次方
        cutRight = 0
        for k in range(pow(2,i)):
            cutLeft = cutRight
            if pow(2,i)-1 == k:
                cutRight = int(Cut[-1])
            else:
                cutRight = int(Cut[k]) + 1
            for j in range(cutLeft, cutRight):
                index = int(X1[j]-1)
                sum_left += timeRobot[index]
                if (sum_left/sum > 0.5):
                    Cut[pow(2,i)+k-1] = j
                    a = sum/2 - (sum_left - timeRobot[index])
                    b = sum_left - sum/2
                    sum_left = 0
                    if (a < b): # 看二分的割点更接近左值还是右值
                        Cut[pow(2,i)+k-1] = j - 1
                    break
        Cut = np.sort(Cut)
        sum = sum/2

    for i in range(len(Cut)-1):
        X1 = np.insert(X1, int(Cut[i])+1+i, 0) # 分割后插零

    return X1

"""
# 以下代码行可用来单独测试decoding.py
file="C:\\Users\\Administrator\\Desktop\\Ask\\data\\MathEx.xlsx"
m1 = excel2m(file,0)
print(searchCut(m1[0],8,29,file),end='\n')
"""


