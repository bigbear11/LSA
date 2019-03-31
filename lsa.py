# -*- coding: utf-8 -*-

import numpy as np
import jieba

class lsa():
    def __init__(self, docs):
        self.docs = []
        self.vocabs = set()
        self.words_dict(docs)

    def words_dict(self, docs):
        for doc in docs:
            doc = doc.strip()
            words = list(filter(lambda x: len(x) > 1, jieba.lcut(doc)))
            self.docs.append(words)
            self.vocabs.update(words)

        self.vocabs = list(self.vocabs)
        self.word2idx = dict(zip(self.vocabs, range(len(self.vocabs))))

    def similar(self):
        k=2
        matrix = self.matrix()

        U, S, V = np.linalg.svd(matrix)

        sort_idx = np.argsort(-U)
        topk = sort_idx[:, 1:k+1]
        print("word \t similarity")
        for widx, word in enumerate(self.vocabs):
            line = word + "\t"
            idxs = topk[widx]
            for idx in idxs:
                line += str(self.vocabs[idx]) + " "
            print(line)
    def matrix(self):
        matrix = np.zeros([len(self.vocabs), len(self.docs)])
        for docidx, words in enumerate(self.docs):
            for word in words:
                matrix[self.word2idx[word], docidx] += 1
        return matrix

        matrix = self.matrix()

        U, S, V = np.linalg.svd(matrix)

        sort_idx = np.argsort(-U)
        topk = sort_idx[:, 1:k+1]
        print("word \t similarity")
        for widx, word in enumerate(self.vocabs):
            line = word + ":\t"
            idxs = topk[widx]
            for idx in idxs:
                line += str(self.vocabs[idx]) + " "
            print(line)

    def topic(self):
        k=1
        matrix = self.matrix()

        U, S, V = np.linalg.svd(matrix)

        sort_idx = np.argsort(-V, axis=1)
        topk = sort_idx[1:k+1, :]
        print(topk)

if __name__ == '__main__':
    doc1 = """2018年4月底上线的贝壳找房，定位以技术驱动的品质居住服务平台，适时推出了VR看房。以“VR看房、VR讲房、VR带看”三大VR核心
    功能。"""

    doc2 = """安居客以“帮助人们实现家的梦想”为企业愿景，全面覆盖新房、二手房、租房、商业地产四大业务，同时为开发商与经纪人提供高效的网络推广平台。2012年，安居客发力移动互联市场，
    旗下“安居客新房”、“安居客二手房”、“安居客租房”三大手机找房APP使用用户突破2500万，占移动找房70%的市场份额"""

    docs = [doc1, doc2]
    model = lsa(docs)
    model.similar()
   # model.topic()
