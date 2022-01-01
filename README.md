# 基于K-means的文本聚类分析


# 一，项目介绍

数据引用：[天池新冠问题数据](https://tianchi.aliyun.com/dataset/dataDetail?spm=5176.12281978.0.0.455f248bNzZ6Nf&dataId=76751)

## 1.原理
> 使用sklearn的K-means算法对文本进行聚类；通过jieba的分词和gensim的word2vec获取词向量，然后设定聚类数量，最后根据聚类的效果排序打印查看

## 2.项目结构
```bash
│  README.md
│  requirements.txt
│  W2V_k-means.py  # k-means聚类分析
│  W2V_train.py  # w2v训练
│
├─data  # 训练数据
│      tianchi_data.csv
│
├─output  # w2v保存
│      model.w2v
```

# 二，使用项目

## 环境

```bash
gensim==4.1.2
jieba==0.42.1
numpy==1.20.1
scikit_learn==1.0.2
```

## 1.下载

`git clone git@github.com:eat-or-eat/text-clustering-analysis.git`

## 2.(选)如果要用自己的数据，需要根据自己的数据内容修改W2V_train.load_data()函数读取数据

## 3.运行

`python ./W2V_k-means.py`

打印示例:

```bash
cluster 1 , avg distance 0.729801: 
清热解毒口服液治疗流感的作用大吗怎么预防流感
咳嗽有痰低烧皮疹头皮按压和太阳穴肿怎么办咳嗽有痰低烧皮疹可以服用阿奇霉素吗
肺结核病人可以输脂肪乳吗？肺癌晚期可以输脂肪乳吗？
喉咙疼痛，清晨吐痰带血怎么办如何治疗喉咙疼痛，清晨吐痰带血
我太太昨天咯血，鲜红的颜色严重吗咯血吃食物是否应该注意什么
坤康清热解毒口服液的效果比较好吗清热解毒口服液和板蓝根一样吗
灰尘大会引发支气管炎吗消炎药可以雾化吗
从视触叩听方面可以说明肺气肿的体征吗视触叩听是指什么
银诺克和力克有什么不一样吗银诺克和力克可以同时服用吗
请问哮喘用免疫介入疗法治疗需要多少钱？免疫介入疗法的中医理论是什么
---------
cluster 4 , avg distance 0.786764: 
请问肺结核服药后起痘痘是什么原因？肺结核服药后起痘痘是药物过敏吗？
请问如何预防和治疗粟粒性肺结核？粟粒性肺结核能治愈吗？
CT能区分肺结核和肺癌的区别吗肺结核和肺癌都需要手术治疗吗
...
```


