import jieba
import gensim
from gensim.models import Word2Vec

# 路径配置
data_pat = './data/tianchi_data.csv'
model_pat = './output/model.w2v'


# 加载数据处理成corpus
def load_data(data_path):
    corpus = []
    with open(data_path, encoding='utf8') as f:
        for line in f:
            line_l = line.split(',')
            sentence = line_l[2] + line_l[3]
            sentence = jieba.lcut(sentence)
            corpus.append(sentence)
    corpus = corpus[1:]  # 去掉行标题
    return corpus


# 训练模型
def train_w2v(data_path, model_path, dim):
    corpus = load_data(data_path)
    model = Word2Vec(sentences=corpus, vector_size=dim, min_count=1)
    model.save(model_path)
    return model


# 查找相似词
def similar_word(model_path):
    model = Word2Vec.load(model_path)
    while True:
        word = input('请输入一个新冠肺炎相关的词:')
        try:
            print(model.wv.most_similar(word))
        except KeyError:
            print('没有找到与这个词相关的词，请重试')


if __name__ == '__main__':
    # 测试用例
    # get = load_data(data_path)
    train_w2v(data_pat, model_pat, 50)
    # similar_word(model_path)
