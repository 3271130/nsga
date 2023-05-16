#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from mpl_toolkits.mplot3d import Axes3D #三维图
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import pandas as pd
from decoding import excel2m, searchCut
from CROSSOVER import check_valid, optimize_chromosome
from TOPOALL import topoall
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import matplotlib;
matplotlib.use('TkAgg')
from tqdm import tqdm # 进度条设置
import time
#导入数据

#====全局变量
#优先矩阵
file = "C:\\Users\\Administrator\\Desktop\\Ask\\data\\MathEx.xlsx"
pm = excel2m(file,3)
#====多目标优化算法一次 只能求解单个时刻的解
class GaMultiobjective(object):  #self是类中的一个
    def __init__(self):
        self.NST = 8 # 固定工作站数量为8个
        self.Na = 29 # BUXEY算例的任务有29个
        self.N = self.Na + self.NST - 1  # 变量个数（待分配的任务个数）
        self.timeRobot_a = excel2m(file,2)[0]
        self.timeRobot_b = excel2m(file,2)[1]
        self.timeRobot_c = excel2m(file,2)[2]

        # =========定义遗传算法的参数、群体=====
        self.NP = 100  # 种群个数
        self.max_gen = 10 # 最大迭代次数
        self.A1 = 80  # 目标函数1的惩罚项,可接受值,这里是CTtheory
        self.A2 = 60  # 目标函数2的惩罚项,可接受值
        self.A3 = 0.99 # 目标函数3的惩罚项,可接受值
        self.Pc = 0.8 # 交叉概率
        self.Pm = 0.2 # 变异概率

        self.bat_gen = int(0.5 * self.max_gen) # 蝙蝠算法的迭代次数
        self.Qmin = 0 # 蝙蝠的声波频率最小值
        self.Qmax = 2 # 蝙蝠的声波频率最大值
        self.A = 0.5 # 蝙蝠的响度
        self.r = 0.5 # 蝙蝠的脉冲频率

    # ===初始化种群====
    def Initialize(self):
        """
        :return: 十进制格式 (“行数:种群个数;列数,任务个数加虚拟基因个数NP+NST-1”的矩阵)
        """
        X = np.zeros((self.NP, self.N))
        originalX = topoall(file, self.NP, self.Na) # 100*29的满足优先关系的数组
        for i in range(self.NP): # 插入虚拟基因
            X[i] = searchCut(originalX[i], self.NST, self.Na, file)
        # 这里写入一份初始种群,后续可能调用
        data = pd.DataFrame(X)
        writer = pd.ExcelWriter("C:\\Users\\Administrator\\Desktop\\Ask\\data\\OutPut.xlsx")
        data.to_excel(writer,'初始种群', header=False, index= False)
        writer.save()
        writer.close()

        print("初始化种群成功\n","种群的第一条染色体是：\n",X[0])
        return X

    # ==========定义目标函数、和对应的惩罚项=============
    # 定义目标函数1：工序节拍时间

    def function1(self, X1):
        """
        X1是染色体吗?是的
        """
        CT=0
        Time_a,Time_b,Time_c,timeMin=0,0,0,0#设置三个计时器
        L0=0 #线平衡率，传给function3
        for i in range(len(X1)):
            index=int(X1[i])
            if(index!=0): # 计算每个工作站选择不同机器人的工作时间
                Time_a+=self.timeRobot_a[index-1]
                Time_b+=self.timeRobot_b[index-1]
                Time_c+=self.timeRobot_c[index-1]
            else: # 选择用时最少的机器人安排在工作站
                if(Time_a<=Time_b):
                    if(Time_c<=Time_a):
                        timeMin=Time_c
                    else:
                        timeMin=Time_a
                else:
                    if(Time_c<=Time_b):
                        timeMin=Time_c
                    else:
                        timeMin=Time_b
                L0+=timeMin # 所有工作站的时间加和
                if(timeMin>CT): # 瓶颈工作站的时间是整个工序的节拍时间
                    CT=timeMin
                Time_a,Time_b,Time_c,timeMin=0,0,0,0 # 计时器清零

        return [CT,L0]

    # 定义目标函数2：均衡系数
    def function2(self, X1):
        L = self.function3(X1)
        Time_a,Time_b,Time_c,Sum_square,timeMin = 0,0,0,0,0 # 设置三个计时器
        for i in range(len(X1)):
            index=int(X1[i])
            if(index!=0):
                Time_a += self.timeRobot_a[index-1]
                Time_b += self.timeRobot_b[index-1]
                Time_c += self.timeRobot_c[index-1]
            else: # 选择用时最少的机器人安排在工作站
                if(Time_a<=Time_b):
                    if(Time_c<=Time_a):
                        timeMin=Time_c
                    else:
                        timeMin=Time_a
                else:
                    if(Time_c<=Time_b):
                        timeMin=Time_c
                    else:
                        timeMin=Time_b
                Sum_square += (timeMin-L/self.NST)**2
                Time_a,Time_b,Time_c,timeMin = 0,0,0,0

        Sum_square=(Sum_square/self.NST)**0.5
        return Sum_square

    # 定义目标函数3：负的线平衡率
    def function3(self, X1):
        listResult = self.function1(X1)       
        L=-listResult[1]/(listResult[0]*self.NST)
        return L

    # 对应的约束：节拍约束(本例子calc_e（） 不起作用，已通过其它方法解决掉该约束）

    def calc_e(self, X1):
        """
        函数1 对应的个体惩罚项
        :param X1: 种群中的染色体
        :return:
        """
        cost = 0
        for i in range (int(len(X1) / self.NST) + 1):
            cost += X1[i]
        return cost

    # =======支线算法：修复染色体==========
    def fix(self, X1):
        """
        :param X1: 种群中的染色体
        :return: X1
        """
        # 解决染色体的问题：
        # a.多0和少0; b.有缺失任务; c.有重复任务; d.存在连续0; e.不满足优先关系
        genes_X1 = set([x for x in X1 if x != 0]) # 取出染色体中的任务基因
        value = 0   
        for x in X1:
            if (x < 0) or (x>self.Na):
                value = 1 
                break # 简单查看任务基因是否超出限制    
        if (len(genes_X1) != self.Na) or (value == 1): # 检查是否有缺失的任务基因
            missing_genes = set(range(1, self.Na + 1)) - genes_X1               
        for i in range(len(X1)): # 替换重复的任务基因
            if X1[i] in genes_X1:
                genes_X1.remove(X1[i])
                if (X1[i] < 0) or (X1[i] > self.Na):
                    if len(missing_genes) != 0:
                        X1[i] = missing_genes.pop()
            else:
                numOf0 = np.sum(X1 == 0)
                if numOf0 > self.NST-1 :
                    X1[i] = missing_genes.pop()
                if numOf0 < self.NST-1 :
                    X1[i] = 0
                # 缺失的任务个数=超限个数+多0个数+重复个数
                if (numOf0 == self.NST-1) and (X1[i] != 0) :
                    X1[i] = missing_genes.pop()

        X1 = [x for x in X1 if (x != 0)and(x <= self.Na)and(x > 0)]
        X1 = searchCut(X1, self.NST, self.Na, file) # 重新插0
        try :
            if check_valid(X1, pm) == 0: # 重新拓扑:使其满足优先关系
                X1=optimize_chromosome(X1, pm , self.Na, self.NP)
        except IndexError:
            print(X1,"eRROR")
        return X1

    # =======遗传算法 交叉操作==========
    def crossover(self, X):
        """
        :param X: 十进制群体(100*36的矩阵)
        :return:交叉后的十进制种群
        """
        """按顺序选择2个个体进行交叉操作,如果交叉部分的虚拟基因个数不相等则不交叉"""
        for i in range(0,X.shape[0],2):
            parent1 = X[i].copy()  # 父亲
            parent2 = X[i + 1].copy()  # 母亲
            child1 = parent1 # 声明第一个娃
            child2 = parent2 # 声明第二个娃
            ret=np.random.randint(0,self.Na-1,2)
            if ret[0] > ret[1]:
                ret[0], ret[1] = ret[1], ret[0]  # 确定交叉点并排序      
            # 虚拟基因的计数器
            numOf0InChild1, numOf0InChild2 = 0, 0
            # 交叉：交叉点之间的部分片段互换
            for j in range(ret[0], ret[1]): # 找，查，删child1虚拟基因
                if (child1[j] == 0):
                    numOf0InChild1 += 1
                    child1 = np.delete(child1, j-numOf0InChild1)
            for j in range(ret[0], ret[1]): # 找，查，删child2虚拟基因
                if (child2[j] == 0):
                    numOf0InChild2 += 1
                    child2 = np.delete(child2, j-numOf0InChild1)
            # 如果交换部分的虚拟基因个数不相等，则取消本次交换
            if (numOf0InChild1 != numOf0InChild2) or (ret[0] == ret[1]):
                continue
            for j in range(ret[0], ret[1]-numOf0InChild1): # 交换
                child1[j] = parent2[j]
                child2[j] = parent1[j]
            for j in range(numOf0InChild1):# 给娃在交换的片段随机插num个虚拟基因
                if ret[1]-ret[0] == 1 :
                    child1 = np.insert(child1, ret[1], 0)
                    child2 = np.insert(child2, ret[1], 0)
                else: # 不可以有“low >= high”
                    child1 = np.insert(child1, np.random.randint(ret[0], ret[1]-numOf0InChild1+j, 1),0)
                    child2 = np.insert(child2, np.random.randint(ret[0], ret[1]-numOf0InChild2+j, 1),0)
            ### 交换完成，交换后的染色体需要进一步调整 ###
            child1 = self.fix(child1)
            child2 = self.fix(child2)
            try:
                X[i, :] = child1
                X[i + 1, :] = child2
            except:
                continue
        print("完成一次交叉操作")

        return X

    # =======遗传算法 变异操作==========

    def mutation(self, X):
        """
        :param X: 十进制群体(100*36的矩阵)
        :return: 变异后的群体 （十进制）
        """
        """变异操作"""
        for i in range(X.shape[0]):
            child = X[i].copy()  # 不影响原排布
            ret=np.random.randint(0,self.Na-1,2)
            if ret[0] > ret[1]:
                ret[0], ret[1] = ret[1], ret[0]  # 确定变异点并排序  
            # 虚拟基因的计数器
            numOf0InChild,t,k = 0,0,0
            num0OfIndex = np.zeros(self.NST-1)
            childmiddle = np.zeros(ret[1]-ret[0]+1)
            for j in range(ret[0], ret[1]): # 找，查，删child虚拟基因
                if (child[j] == 0):
                    num0OfIndex[t]=j
                    numOf0InChild += 1
                    child = np.delete(child, j)
                    t += 1
                else:
                    childmiddle[k] = child[j]
                    k += 1   
            random.shuffle(childmiddle) # 打乱变异点之间的基因顺序
            k=0 # 计数器清零
            for j in range(ret[0], ret[1]-numOf0InChild):
                child[j] = childmiddle[k]
                k += 1
            t=0 # 计数器清零
            for j in range(numOf0InChild):
                child = np.insert(child, int(num0OfIndex[t]), 0)
                t += 1
            # 判断孩子是否满足优先关系，不满足则尽量不破坏原染色体调整
            if check_valid(child, pm) == 0:
                child = optimize_chromosome(child, pm, self.Na, self.NP)
            try:
                X[i, :] = child
            except:
                continue
        print("完成一次变异操作")

        return  X
    
    # ===选择函数:嵌入到快速非支配排序中 ====
    def update_best(self, p_fitness, p_e, q_fitness, q_e, A):
        """
        :param p_fitness: 个体的p的适应度
        :param p_e:  个体的p的惩罚项
        :param q_fitness: 个体的q的适应度
        :param q_e: 个体的q的惩罚项
        :param A :可接受的惩罚项阈值
        :return: 如果个体p好,返回1;个体q好,返回-1;两者一样,返回0。
        """
        # 规则1，如果 p 和 q都没有违反约束，则取适应度小的。两者一样返回0
        if p_e <= A and q_e <= A:
            if p_fitness < q_fitness:
                return 1  # p好
            elif p_fitness == q_fitness:
                return 0  # 两者一样
            else:
                return -1  # q好
        # 规则2，如果q违反约束而p没有违反约束,则取p
        if p_e < A and q_e >= A:
            return 1  # p好
        # 规则3，如果p违反约束而q没有违反约束,则取q
        if p_e >= A and q_e < A:
            return -1  # q好
        # 规则4，如果两个都违反约束,则取适应度值小的
        if p_fitness <= q_fitness:
            return 1  # p好
        else:
            return -1  # q好

    # ====多目标优化：快速非支配排序=====
    def fast_non_dominated_sort(self, values, coste):
        """
        优化问题一般是求最小值
        由于复杂约束含有惩罚项，所以这里的快速非支配排序 ，较常规快速非支配排序 有所不同
        :param values: 解集【目标函数1解集,目标函数2解集...】
        :param coste: 惩罚项【目标函数1惩罚项,目标函数2惩罚项...】
        :return:返回解的各层分布集合序号。类似[[1], [9], [0, 8], [7, 6], [3, 5], [2, 4]] 其中[1]表示Pareto 最优解对应的序号
        """

        values11 = values[0]  # 函数解集
        S = [[] for i in range(0, len(values11))]  # 存放 每个个体支配解的集合。
        front = [[]]  # 存放群体的级别集合，一个级别对应一个[]
        n = [0 for i in range(0, len(values11))]  # 每个个体被支配解的个数 。即针对每个解，存放有多少好于这个解的个数
        rank = [np.inf for i in range(0, len(values11))]  # 存放每个个体的级别

        for p in range(0, len(values11)):  # 遍历每一个个体
            # ====得到各个个体 的被支配解个数 和支配解集合====
            S[p] = []  # 该个体支配解的集合 。即存放差于该解的解
            n[p] = 0  # 该个体被支配的解的个数初始化为0  即找到有多少好于该解的 解的个数
            for q in range(0, len(values11)):  # 遍历每一个个体
                less = 0  # 比p个体好的 目标函数值个数
                equal = 0  # p和q一样好 目标函数值个数
                greater = 0  # 比p个体 差的 目标函数值个数
                for k in range(len(values)):  # 遍历每一个目标函数、惩罚项
                    if self.update_best(values[k][p], coste[k][p], values[k][q], coste[k][q], self.A1) == -1:
                        less = less + 1  # q比p 好
                    if self.update_best(values[k][p], coste[k][p], values[k][q], coste[k][q], self.A1) == 0:
                        equal = equal + 1  # (q和p一样好）
                    if self.update_best(values[k][p], coste[k][p], values[k][q], coste[k][q], self.A1) == 1:
                        greater = greater + 1  # q比p 差

                if (less + equal == len(values)) and (equal != len(values)):
                    n[p] = n[p] + 1  # q比p,  比p好的个体个数加1
                elif (greater + equal == len(values)) and (equal != len(values)):
                    S[p].append(q)  # q比p差，存放比p差的个体解序号
            # =====找出Pareto 最优解，即n[p]===0 的 个体p序号。=====
            if n[p] == 0:
                rank[p] = 0  # 序号为p的个体，等级为0即最优
                if p not in front[0]:
                    # 如果p不在第0层中
                    # 将其追加到第0层中
                    front[0].append(p)  # 存放Pareto 最优解序号

        # =======划分各层解========
        i = 0
        while (front[i] != []):  # 如果分层集合为不为空
            Q = []
            for p in front[i]:  # 遍历当前分层集合的各个个体p
                for q in S[p]:  # 遍历p 个体 的每个支配解q
                    n[q] = n[q] - 1  # 则将fk中所有给对应的个体np-1
                    if (n[q] == 0):
                        # 如果nq==0
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)  # 存放front=i+1 的个体序号

            i = i + 1  # front 等级+1
            front.append(Q)

        del front[len(front) - 1]  # 删除循环退出 时 i+1产生的[]

        return front  # 返回各层 的解序号集合 # 类似[[1], [9], [0, 8], [7, 6], [3, 5], [2, 4]]

    # =============多目标优化：拥挤距离================

    def crowding_distance(self, values, front, popsize):
        """
        :param values: 群体[目标函数值1,目标函数值2,...]
        :param front: 群体解的等级,类似[[1], [9], [0, 8], [7, 6], [3, 5], [2, 4]]
        :param  popsize: 当前群体个数
        :return: front 对应的拥挤距离[[],[],[],[]]
        """
        distance = np.zeros(shape=(popsize,))  # 拥挤距离初始化为0
        for rank in front:  # 遍历每一层Pareto解rank为当前等级
            for i in range(len(values)):  # 遍历每一层函数值(先遍历群体函数值1，再遍历群体函数值2...)
                valuesi = [values[i][A] for A in rank]  # 取出rank等级对应的目标函数值i集合
                rank_valuesi = zip(rank, valuesi)  # 将rank,群体函数值i集合在一起
                sort_rank_valuesi = sorted(rank_valuesi, key=lambda x: (x[1], x[0]))  # 先按函数值大小排序，再按序号大小排序

                sort_ranki = [j[0] for j in sort_rank_valuesi]  # 排序后当前等级rank
                sort_valuesi = [j[1] for j in sort_rank_valuesi]  # 排序后当前等级对应的群体函数值i
                distance[sort_ranki[0]] = np.inf  # rank等级中的最优解距离为inf
                distance[sort_ranki[-1]] = np.inf  # rank等级中的最差解距离为inf

                # 计算rank等级中，除去最优解、最差解外。其余解的拥挤距离
                for j in range(1, len(rank) - 2):
                    distance[sort_ranki[j]] = distance[sort_ranki[j]] + (sort_valuesi[j + 1] - sort_valuesi[j - 1]) / (max(sort_valuesi) - min(sort_valuesi))  # 计算距离

        # 按照格式存放distances
        distanceA = [[] for i in range(len(front))]
        for j in range(len(front)):  # 遍历每一层Pareto解rank为当前等级
            for i in range(len(front[j])):  # 遍历给rank等级中每个解的序号
                distanceA[j].append(distance[front[j][i]])

        return distanceA

    # =============多目标优化：精英选择================

    def elitism(self, front, distance, solution):
        """
        精英选择策略
        :param front: 父代与子代组合构成的解的等级
        :param distance:  父代与子代组合构成的解 拥挤距离
        :param solution:  父代与子代组合构成的解
        :return:  返回群体解。群体数量=（父代+子代）//2
        """
        X1index = []  # 存储群体编号
        pop_size = len(solution) // 2  # 保留的群体个数即:(父辈+子辈)//2
        # pop_size=self.NP

        for i in range(len(front)):  # 遍历各层
            rank_distancei = zip(front[i], distance[i])  # 当前等级与当前拥挤距离的集合
            sort_rank_distancei = sorted(rank_distancei, key=lambda x: (x[1], x[0]),
                                         reverse=True)  # 先按拥挤距离大小排序，再按序号大小排序,逆序
            sort_ranki = [j[0] for j in sort_rank_distancei]  # 排序后当前等级rank
            sort_distancei = [j[1] for j in sort_rank_distancei]  # 排序后当前等级对应的 拥挤距离i

            if (pop_size - len(X1index)) >= len(sort_ranki):  # 如果X1index还有空间可以存放当前等级i 全部解
                X1index.extend([A for A in sort_ranki])

            # print('已存放len(X1index)', len(X1index))
            # print('当前等级长度', len(sort_ranki))
            # print('需要存放的总长度,popsize)
            # num = pop_size-len(X1index)# X1index 还能存放的个数
            elif len(sort_ranki) > (pop_size - len(X1index)):  # 如果X1空间不可以存放当前等级i 全部解
                num = pop_size - len(X1index)
                X1index.extend([A for A in sort_ranki[0:num]])
        X1 = [solution[i] for i in X1index]

        return X1

    # =============蝙蝠算法：初始种群迭代=============
    def bat(self, X):
        """
        我是一只蝙蝠！
        :param X: 当前的种群
        :return: 返回群体。群体数量 = self.NP
        """
        Q = np.zeros((self.NP, 1)) # 记录种群中蝙蝠的频率
        v = np.zeros((self.NP, self.N)) # 记录种群中蝙蝠的速度
        S = np.zeros((self.NP, self.N)) # 记录种群中蝙蝠的更新后的位置
        Fitness = np.zeros((self.NP, 1)) # 记录种群中蝙蝠的适应度(这里选择节拍的倒数)
        # 找到种群中的初始最优解
        best = np.zeros((1, self.N))
        for i in range(X.shape[0]):
            fmax = 0.0
            Fitness[i] = 1.0/(self.function1(X[i])[0])
            if Fitness[i] > fmax:
                best, fmax = X[i], Fitness[i]
        # 循环迭代种群里的所有蝙蝠
        for i in range(self.NP):
            Q[i] = np.random.randint(self.Qmin, self.Qmax) # 第i条染色体的声波频率
            v[i] = v[i] + (X[i] - best) * Q[i]
            S[i] = X[i] + v[i]
            # 大于蝙蝠的脉冲频率就走一步
            if random.random() > self.r:
                # 因子1限制了蝙蝠随机行走的步长
                S[i] = best + 1 * np.random.randint(-2, high=2, size=(1, self.N)) # 生成一个局部解
            # 评估新的解
            S[i] = self.fix(S[i])
            Fnew = 1.0/(self.function1(X[i])[0])
            # 用生成的更优解取代原来的解
            if (Fnew > Fitness[i]) and (random.random() < self.A):
                X[i] = S[i]
                Fitness[i] = Fnew
            # 更新种群中蝙蝠最高的适应度值
            if Fnew > fmax:
                best = S[i]
                fmax = Fnew
        # 此时的种群满足优先关系
        print("完成一次蝙蝠种群的更新")

        return X

    # =============遗传算法多目标优化主函数=============
    def main(self):
        parenten = self.Initialize()  #  初始种群f.shape(NP,Na+Nst-1)
        for i in tqdm(range(self.bat_gen), desc = 'It\'s a bat'): 
            parenten = self.bat(parenten)
        for i in tqdm(range(self.max_gen - self.bat_gen), desc = 'It\'s a chorosome'):
            # =====1.对父代进行 交叉、变异 得到子代 X3============
            if random.random() <= self.Pc:
                X1 = self.crossover(parenten)  # 以一定概率交叉
            else:
                X1 = parenten
            if random.random() <= self.Pm:
                X2 = self.mutation(X1)  # 以一定概率变异
            else:
                X2 = X1

            # ======2.父代与子代合并==========
            parentchildten = np.concatenate([parenten , X2], axis=0)

            # ====3.计算合并群体的目标函数1、目标函数2... 值
            # 函数1 对应的目标函数值
            values1 = np.zeros(shape=len(parentchildten), )
            for i in range(len(parentchildten)):  # 遍历每一个个体
                values1[i] = self.function1(parentchildten[i])[0]
            # 函数2 对应的目标函数值
            values2 = np.zeros(shape=len(parentchildten), )
            for i in range(len(parentchildten)):  # 遍历每一个个体
                values2[i] = self.function2(parentchildten[i])

            # 函数3 对应的目标函数值
            values3 = np.zeros(shape=len(parentchildten), )
            for i in range(len(parentchildten)):  # 遍历每一个个体
                values3[i] = self.function3(parentchildten[i])
            values = [values1, values2, values3]  # 合并群体目标函数值

            # ====4.计算 合并群体的目标函数1惩罚项、目标函数2惩罚项... 值
            # 由于这几个目标函数的惩罚项是一样的，没有单独的惩罚项
            coste1 = np.zeros(shape=len(parentchildten), )
            for i in range(len(parentchildten)):  # 遍历每一个个体
                coste1[i] = self.calc_e(parentchildten[i])
            coste = [coste1, coste1, coste1]  # 合并群体惩罚项值 (所以惩罚项为一样的）

            # ====5.对合并群体进行快速非支配排序(考虑惩罚项的快速非支配排序）====
            front = self.fast_non_dominated_sort(values, coste)  # front 为各个解的等级
            # ====6.对合并群体进行【拥挤距离】计算========
            distanceA = self.crowding_distance(values, front, 2 * self.NP)  # 2*self.NP 因为是父代加子代,所以是2倍群体
            # =====7.对于合并群体进行【精英选择】====
            X3 = self.elitism(front, distanceA, parentchildten)
            # ===将精英选择得到的一半最优秀个体替换为父辈
            parentten = np.array(X3)
            parentten = parentten.reshape(self.NP, self.N)

        # =====8.迭代结束后的群体为 parentten==
        # 函数1 对应的目标函数值
        values1 = np.zeros(shape=len(parentten), )
        for i in range(len(parentten)):  # 遍历每一个个体
            values1[i] = self.function1(parentten[i])[0]
        # 函数2 对应的目标函数值
        values2 = np.zeros(shape=len(parentten), )
        for i in range(len(parentten)):  # 遍历每一个个体
            values2[i] = self.function2(parentten[i])
        # 函数3 对应的目标函数值
        values3 = np.zeros(shape=len(parentten), )
        for i in range(len(parentten)):  # 遍历每一个个体
            values3[i] = self.function3(parentten[i])
        values = [values1, values2, values3]  # 合并群体目标函数值
        #  对应的惩罚项
        coste1 = np.zeros(shape=len(parentten), )
        for i in range(len(parentten)):  # 遍历每一个个体
            coste1[i] = self.calc_e(parentten[i])
        coste = [coste1, coste1, coste1]  # 合并群体惩罚项值

        # =====9.对迭代结束后的群体进行快速非支配排序==
        front = self.fast_non_dominated_sort(values, coste)  # front 为各个解的等级
        jie = np.zeros(shape=(len(front[0]), self.N))
        for j in range(len(front[0])):  # 遍历第i层各个解
            jie[j] = parentten[front[0][j]]

        # ====10，打印结果============
        print(front[0])  # 打印最优等级
        print('最优解个数', len(front[0]))
        print('最优十进制解', jie)

        fig = plt.figure(figsize=(6, 6))  # 指定画布大小
        ax = fig.gca(projection='3d')  # 指定为3D
        ax.scatter(values1, values2, values3)
        ax.set_xlabel('生产节拍时间/s')
        ax.set_ylabel('均衡系数/none')
        ax.set_zlabel('负的线平衡率/%')
        plt.ion()
        plt.pause(10)
        plt.close()
        # plt.show()


if __name__ == "__main__":
    start = time.time()

    ga1 = GaMultiobjective()
    ga1.main()

    end = time.time()
    running_time = end - start - 10 # 减去图像展示时间
    print('time cost : %.5f sec' %running_time)

