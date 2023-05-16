import networkx as nx
import pandas as pd
import itertools
import numpy as np

def topoall(file,NP,Na):
    # 读取excel文件
    df = pd.read_excel(file, sheet_name=3, header=None, index_col=None)
    # 创建有向图
    G = nx.DiGraph()
    # 添加节点
    for node in df.index:
        G.add_node(node)
    # 添加边
    for i, row in df.iterrows():
        for j, value in row.iteritems():
            if value == 1:
                G.add_edge(i, j)

    # 计算所有拓扑排序(取前1000种)
    topological_sorts = list(itertools.islice(nx.all_topological_sorts(G),1000))
    X = np.zeros((NP, Na))
    suiji = np.random.randint(low=0, high=1000, size=(1,NP))
    jia1 = np.ones((NP, Na))
    for i in range(NP):
        index = int(suiji[0][i])
        X[i] = topological_sorts[index]
    X = X + jia1

    return X

"""
# 以下代码行可用来单独测试TOPOALL.py
file = "C:\\Users\\Administrator\\Desktop\\Ask\\data\\MathEx.xlsx"
print(topoall(file,10,29))
"""