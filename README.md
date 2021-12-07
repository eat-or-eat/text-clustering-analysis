# text-clustering-analysis
通过word2vec实现文本向量化，然后用k-means算法进行分类，实现无监督的数据聚类分析

# 一，使用项目

## 环境

```bash
gensim==4.1.2
jieba==0.42.1
numpy==1.18.5
scikit_learn==0.23.1
```



## 1.下载

`git clone git@github.com:eat-or-eat/text-clustering-analysis.git`

## 2.(选)如果要用自己的数据，需要根据自己的数据内容修改W2V_train.load_data()函数读取数据

## 3.运行

`python ./W2V_k-means.py`

打印示例:

```bash
cluster 0:
红霉素肠溶胶囊肺炎吃几次红霉素肠溶胶囊肺炎每日吃几次
请问利巴韦林能和优卡丹一起吃吗利巴韦林能和优卡丹不能一起吃
...
----------
cluster 1:
请问肺结核服药后起痘痘是什么原因？肺结核服药后起痘痘是药物过敏吗？
请问如何预防和治疗粟粒性肺结核？粟粒性肺结核能治愈吗？
...
----------
cluster 2:
请问肺结核的复发几率有多大？肺结核病有什么特殊治疗方法？
肺气肿吃什么药比较有效果氨茶碱可以治疗肺气肿吗
肺气肿患者接着抽烟会造成什么后果抽烟对孩子有什么影响
...
----------
cluster 3:
病毒性肺炎诊断标准是什么？病毒性肺炎用什么药
患有病毒性肺炎能治愈吗？病毒性肺炎有哪些症状？
...
----------
...
```

# 二，项目介绍说明

```bash
|----data\  # 数据（语料）文件夹
|    |----tianchi_data.csv
|----output\  # 输出模型文件夹
|    |----model.w2v
|----README.md
|----requirements.txt
|----W2V_k-means.py  # kmeans聚类打印
|----W2V_train.py  # 训练词向量
```



```
引用：天池数据
https://tianchi.aliyun.com/dataset/dataDetail?spm=5176.12281978.0.0.455f248bNzZ6Nf&dataId=76751
```

