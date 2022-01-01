import time
import jieba

from gensim.models import Word2Vec

"""
该脚本用于训练词向量，供下游任务使用
流程：读取csv数据->训练word2vec->保存model.w2v权重
"""

# 路径配置
data_path = './data/tianchi_data.csv'
model_path = './output/model.w2v'


def load_data(data_path):
    """
    从语料路径加载语料数据，返回corpus
    :param data_path: 语料相对路径
    :return: 返回分词后的嵌套list
    """
    corpus = []
    with open(data_path, encoding='utf8') as f:
        for line in f:
            line_l = line.split(',')
            sentence = line_l[2] + line_l[3]
            sentence = jieba.lcut(sentence)
            corpus.append(sentence)
    corpus = corpus[1:]  # 去掉行标题
    return corpus


def train_w2v(data_path, model_path, dim):
    """
    封装了加载数据，训练模型与保存模型的函数，返回模型结果
    :param data_path: 语料相对路径
    :param model_path: 保存模型的相对路径
    :param dim: 词向量维度
    :return: 返回模型
    """
    corpus = load_data(data_path)
    model = Word2Vec(sentences=corpus, vector_size=dim, min_count=1)  # 最小出现次数为1
    model.save(model_path)
    return model


def similar_word(model_path):
    """
    查找相似词（简单看看模型效果）
    :param model_path: 模型相对路径
    :return: None
    """
    model = Word2Vec.load(model_path)
    while True:
        word = input('请输入一个新冠肺炎相关的词:')
        try:
            print(model.wv.most_similar(word))
        except KeyError:
            print('没有找到与这个词相关的词，请重试')


if __name__ == '__main__':
    # 测试用例
    # corpus = load_data(data_path)

    start_time = time.time()
    model = train_w2v(data_path, model_path, 128)
    print('模型训练耗时：%fs' % (time.time() - start_time))

    # similar_word(model_path)