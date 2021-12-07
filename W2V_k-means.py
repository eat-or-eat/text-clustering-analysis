import jieba
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from collections import defaultdict

from W2V_train import load_data, data_pat, model_pat

n_clusters = 6  # 聚类数量

# 将文本处理成向量，返回np.array数组
def corpus_to_vec(corpus, model):
    vectors = []
    for words in corpus:
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                pass
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    corpus = load_data(data_pat)
    model = Word2Vec.load(model_pat)
    vectors = corpus_to_vec(corpus, model)
    kmeans = KMeans(n_clusters, random_state=0)
    kmeans.fit(vectors)
    sentence_label = defaultdict(list)
    for sentence, label in zip(corpus, kmeans.labels_):
        sentence_label[label].append(''.join(sentence))
    for i_label in range(n_clusters):
        print('cluster %s:' % i_label)
        for i in range(min(10, len(sentence_label[i_label]))):
            print(sentence_label[i_label][i])
        print('----------')


if __name__ == '__main__':
    main()
