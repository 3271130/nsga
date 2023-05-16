import networkx as nx
import random
import decoding
import numpy as np
from decoding import excel2m

def check_valid(child, priority_table): #检查染色体是否满足优先关系和工序紧前约束
    """
    :param child: 交叉后的子代
    :param priority_table: 优先关系矩阵
    :return: 是否满足要求
    """
    # 检查优先关系
    for i in range(len(child) - 1):
        if child[i] != 0 and child[i + 1] != 0:  # 排除虚拟基因
            task1 = int(child[i] - 1)
            task2 = int(child[i + 1] - 1)
            if priority_table[task1][task2] == 0:
                return False
    # 检查工序紧前约束
    for i in range(len(child) - 1):
        if child[i] != 0 and child[i + 1] != 0:  # 排除虚拟基因
            task1 = int(child[i] - 1)
            task2 = int(child[i + 1] - 1)
            if task1 > task2:  # 保证先检查较小的任务
                task1, task2 = task2, task1
            if priority_table[task1][task2] != 0 and child.index(task1 + 1) > child.index(task2 + 1):
                return False
    return True

def optimize_chromosome(chromosome, priorities_matrix, Na, NP):
    # 创建有向图
    chromosome_real=np.zeros(Na)
    G = nx.DiGraph()
    # 添加节点
    i=0
    for gene in chromosome:
        if gene != 0:
            chromosome_real[i] = gene
            G.add_node(gene)
            i += 1
        if i == Na:
            break
    # 添加边
    for i in range(len(chromosome_real)-1):
        if chromosome_real[i] != 0 and chromosome_real[i+1] != 0:
            if priorities_matrix[int(chromosome_real[i]-1)][int(chromosome_real[i+1]-1)] == 1:
                G.add_edge(chromosome_real[i], chromosome_real[i+1])
    # 拓扑排序
    try:
        sorted_chromosome = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # 如果无法进行拓扑排序，直接返回原染色体
        return chromosome
    # 重新排序
    new_chromosome = [0]*len(chromosome_real)
    for i in range(len(sorted_chromosome)):
        new_chromosome[i] = int(sorted_chromosome[i])
    # 将虚拟基因0移动到原位置
    ending=[]
    for item, value in enumerate(chromosome):
        if value==0:
            ending.append(item)
    for i in range(len(ending)):
        new_chromosome.insert(ending[i], 0)
    # 计算破坏程度
    damage = 0
    for i in range(len(chromosome)-1):
        if chromosome[i] != 0 and chromosome[i+1] != 0:
            try:
                if (chromosome[i] not in sorted_chromosome[:sorted_chromosome.index(chromosome[i+1])]) and (chromosome[i+1] not in sorted_chromosome[:sorted_chromosome.index(chromosome[i])] ):
                    damage += 1
            except:
                    damage = 100000
                    print("这里有一处异常", new_chromosome)
    # 如果破坏程度为15，返回新染色体，否则随机选择两个位置交换后返回
    if damage < 15:
        return new_chromosome
    else:
        # 否则返回初始染色体的随机一个
        num = np.random.randint(0,NP-1)
        new_chromosome = excel2m("C:\\Users\\Administrator\\Desktop\\Ask\\data\\OutPut.xlsx",0)[num]
        return new_chromosome

"""
# 以下代码行可用来单独测试CROSSOVER.py
pm=decoding.excel2m("C:\\Users\\Administrator\\Desktop\\Ask\\data\\MathEX.xlsx",3)
a=[1,2,3,4,0,5,6,8,7,9,0,10,11,0,12,13,14,0,15,16,17,18,0,19,20,21,0,22,23,24,0,25,26,27,28,29];
if check_valid(a,pm)==0:
    print(check_valid(a,pm))
    print(optimize_chromosome(a,pm,29,100))
"""