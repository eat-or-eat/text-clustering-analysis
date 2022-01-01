import jieba
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from collections import defaultdict
from W2V_train import load_data

"""
用于实现K-means距离并用类内距离选取合适的聚类数量
流程：载入数据与词向量->训练模型->计算类内距离->打印查看
"""
# 路径配置
data_path = './data/tianchi_data.csv'
model_path = './output/model.w2v'

n_clusters = 10  # 聚类数量


def corpus_to_vec(corpus, model):
    """
    通过词向量相加的方式获取句向量
    :param corpus: 语料数组
    :param model: 词向量模型
    :return: 句向量
    """
    vectors = []
    for words in corpus:
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                pass
        vectors.append(vector / len(words))
    return vectors


# 欧式距离
def eculid_distance(vec1, vec2):
    """
    计算欧氏距离判断聚类效果
    """
    return np.sqrt((np.sum(np.square(vec1 - vec2))))


def main():
    # 载入数据与词向量
    corpus = load_data(data_path)
    model = Word2Vec.load(model_path)
    vectors = corpus_to_vec(corpus, model)
    # 训练模型
    kmeans = KMeans(n_clusters, random_state=0)
    kmeans.fit(vectors)

    # 计算类内距离
    eu_dis = defaultdict(list)
    for vector_index, label in enumerate(kmeans.labels_):
        vector = vectors[vector_index]  # 某句话的向量
        center = kmeans.cluster_centers_[label]  # 对应的类别中心向量
        distance = eculid_distance(vector, center)  # 计算距离
        eu_dis[label].append(distance)  # 添加
    for label, distance_list in eu_dis.items():
        eu_dis[label] = np.mean(distance_list)  # 对于每一类，将类内所有文本到中心的向量余弦值取平均
    # 打印查看
    label_dis_list = sorted(eu_dis.items(), key=lambda x: x[1])  # 按照平均距离排序，欧式距离越小，相似度越高
    label_sentence_dict = defaultdict(list)
    for sentence_list, label in zip(corpus, kmeans.labels_):  # 取出句子和标签
        label_sentence_dict[label].append(sentence_list)  # 同标签的放到一起
    for label, distance_avg in label_dis_list:  # 按照欧氏距离升序查看
        print("cluster %s , avg distance %f: " % (label, distance_avg))
        sentences = label_sentence_dict[label]
        for i in range(min(10, len(sentences))):
            print(''.join(sentences[i]))
        print("---------")


if __name__ == '__main__':
    main()
