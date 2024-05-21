import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jieba
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nxcom
import re

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 500)

def data_preprocess():
    # 1. 读取文件
    df = pd.read_csv("D:\data\\firm_bus_text.csv",
                     usecols=["Symbol", "EndDate", "BusinessScope", 'LISTINGDATE'])
    df['Symbol'] = df['Symbol'].astype(str).str.zfill(6)


    # 2. 从EndDate中提取年份并筛选数据
    df["Year"] = pd.to_datetime(df["EndDate"]).dt.year
    df = df[(pd.to_datetime(df['LISTINGDATE']) < pd.to_datetime("2020-12-31"))]
    df = df[df["Year"].between(2001, 2022)]
    df = df[['Symbol', 'EndDate', 'BusinessScope']]
    # df["bus_len"]=df["BusinessScope"].str.len()
    df.dropna(inplace=True)

    # df = df[df["bus_len"]>=10]
    # print(df.groupby('Year')['Symbol'].agg(['count']))

    # # 3. 读取fraudnews_count.csv并获取非重复的Symbol值
    # fraud_symbols = pd.read_csv("fraudnews_count.csv")["Symbol"].astype(str).str.zfill(6).unique().tolist()
    # df = df[df["Symbol"].isin(fraud_symbols)]
    # # df.to_csv('temp.csv', encoding='utf-8-sig')
    # # exit()

    # 分词
    def remove_non_chinese(string):
        if not isinstance(string, str):
            return ''
        return re.sub(r'[^\u4e00-\u9fa5]', '', string)

    df['BusinessScope_clear'] = df['BusinessScope'].apply(remove_non_chinese)
    with open('../stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    jieba.load_userdict("../terms/Finwords.txt")
    df['BusinessScope_cut'] = df['BusinessScope_clear'].apply(
        lambda x: " ".join([word for word in jieba.cut(x) if word not in stopwords]))
    df.to_csv('data/BusinessScope_cut.csv', index=False, encoding='utf-8-sig')

def bus_sim_gen():
    # 使用doc2vec技术将BusinessScope列转化为向量
    df = pd.read_csv('data/BusinessScope_cut.csv')
    df["Year"] = pd.to_datetime(df["EndDate"]).dt.year
    df['BusinessScope_cut'] = df['BusinessScope_cut'].astype(str)
    # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df["BusinessScope_cut"])]
    # model = Doc2Vec(documents, vector_size = 300, window = 5, min_count = 1, workers = 4)
    # model.save('model/doc2vec_bus')
    # print("模型已保存")
    model = Doc2Vec.load('model/doc2vec_bus')
    df["Vector"] = df["BusinessScope_cut"].apply(lambda x: model.infer_vector(x.split()))

    # 5. 计算余弦相似度
    results = []
    for year in tqdm(df["Year"].unique()):
        results = []
        yearly_data = df[df["Year"] == year]
        for i in range(len(yearly_data)):
            for j in range(i+1, len(yearly_data)):
                sim = cosine_similarity([yearly_data.iloc[i]["Vector"]], [yearly_data.iloc[j]["Vector"]])[0][0]
                results.append({
                    "Symbol1": yearly_data.iloc[i]["Symbol"],
                    "Symbol2": yearly_data.iloc[j]["Symbol"],
                    "Year": year,
                    "Similarity": sim
                })
        sim_df = pd.DataFrame(results)
        # 6. 保存到bus_sim.csv
        sim_df.to_csv("data/bus_sim"+str(year)+".csv", index=False)

def bus_sim_discr():

    folder_path = 'bus_sim'

    result_data = []

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            if 'Similarity' in df.columns:
                desc_stats = df['Similarity'].describe()
                count_gt_0 = (df['Similarity'] > 0).sum()
                count_gt_0_1 = (df['Similarity'] > 0.5).sum()
                firm_count = df['Symbol1'].nunique()

                result_data.append({
                    'Filename': filename,
                    '#firm': firm_count,
                    'Count (> 0)': count_gt_0,
                    'Count (> 0.5)': count_gt_0_1,
                    **desc_stats
                })
    result_df = pd.DataFrame(result_data)

    result_csv_path = 'bus_sim_discr.csv'
    result_df.to_csv(result_csv_path, index=False)

    print('The discription of business similarity has been saved to ', result_csv_path)

def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1
def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0
def get_color(i, r_off=1, g_off=1, b_off=1):
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)

def panel_to_matrix(data):
    symbols = sorted(set(data['Symbol1']).union(set(data['Symbol2'])))
    # 创建相似度矩阵
    similarity_matrix = pd.DataFrame(index=symbols, columns=symbols)
    similarity_matrix = similarity_matrix.fillna(0)  # 初始化相似度矩阵为0
    # 填充相似度矩阵
    for index, row in data.iterrows():
        symbol1 = row['Symbol1']
        symbol2 = row['Symbol2']
        similarity = row['Similarity']
        similarity_matrix.loc[symbol1, symbol2] = similarity
        similarity_matrix.loc[symbol2, symbol1] = similarity  # 因为是对称矩阵，所以同时填充对角线另一侧

    return similarity_matrix

def bus_net():
    # data = pd.read_csv('../data/bus_sim2001.csv', converters = {'Symbol1':str, 'Symbol2':str})
    # G = nx.Graph()
    #
    # for index, row in data.iterrows():
    #     symbol1 = row['Symbol1']
    #     symbol2 = row['Symbol2']
    #     similarity = row['Similarity']
    #     if similarity > 0.5:
    #         G.add_edge(symbol1, symbol2)

    df = pd.read_csv('../data/bus_sim2004.csv', converters={'Symbol1': str, 'Symbol2': str})
    df = panel_to_matrix(df)
    nodes = range(len(df.columns))
    sim_mat = np.asarray(df)

    G = nx.Graph()
    G.add_nodes_from(nodes)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if (sim_mat[i][j] > 0.5):
                G.add_edge(i, j)

    communities = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)
    set_node_community(G, communities)
    set_edge_community(G)
    node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]
    # external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
    internal_color = [get_color(G.edges[e]['community']) for e in internal]
    pos = nx.spring_layout(G, k=0.5, pos=nx.circular_layout(G))

    nx.draw_networkx(G, pos=pos, node_size=1, with_labels=False, edgelist=internal, alpha=0.3, width=0.8,
                     node_color=node_color, edge_color=internal_color)
    # nx.draw_networkx(G, pos=pos, node_size=0, with_labels=False, edgelist=external, alpha=0.3, edge_color="#333333",
    #                  width=0.8)

    plt.rcParams.update({'figure.figsize': (20, 20)})
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # data_preprocess()
    # bus_sim_gen()
    # bus_sim_discr()
    bus_net()