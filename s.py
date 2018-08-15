#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
import jieba
# import docx2txt
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
# import shutil
# import codecs
# from sklearn import metrics
import pickle
import re


# 保存至文件
def savefile(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)


# 读取文件
def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


# 保存bunch对象
def writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


# 读取bunch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


def corpus_segment(corpus_path, seg_path):
    '''
    corpus_path是未分词语料库路径
    seg_path是分词后语料库存储路径
    '''
    catelist = os.listdir(corpus_path)  # 获取corpus_path下的所有子目录
    '''
    其中子目录的名字就是类别名，例如：
    train_corpus/art/21.txt中，'train_corpus/'是corpus_path，'art'是catelist中的一个成员
    '''
    print("正在分词，请稍候")
    # 获取每个目录（类别）下所有的文件
    for mydir in catelist:
        '''
        这里mydir就是train_corpus/art/21.txt中的art（即catelist中的一个类别）
        '''
        class_path = corpus_path + mydir + "/"  # 拼出分类子目录的路径如：train_corpus/art/
        seg_dir = seg_path + mydir + "/"  # 拼出分词后存贮的对应目录路径如：train_corpus_seg/art/
        if not os.path.exists(seg_dir):  # 是否存在分词目录，如果没有则创建该目录
            os.makedirs(seg_dir)
        file_list = os.listdir(class_path)  # 获取未分词语料库中某一类别中的所有文本
        '''
        train_corpus/art/中的
        21.txt,
        22.txt,
        23.txt
        ...
        file_list=['21.txt','22.txt',...]
        '''
        for file_path in file_list:  # 遍历类别目录下的所有文件
            fullname = class_path + file_path  # 拼出文件名全路径如：train_corpus/art/21.txt
            content = readfile(fullname)  # 读取文件内容
            '''
            此时，content里面存贮的是原文本的所有字符，例如多余的空格、空行、回车等等，
            接下来，我们需要把这些无关痛痒的字符统统去掉，变成只有标点符号做间隔的紧凑的文本内容
            '''
            content1 = content.decode('utf-8')
            content1 = re.sub("[^\u4e00-\u9fa5]", "", content1)  # unicode中文范围
            content = content1.encode('utf-8')
            content_seg = jieba.cut(content)  # 为文件内容分词
            savefile(seg_dir + file_path, ' '.join(content_seg).encode('utf-8'))  # 将处理后的文件保存到分词后语料目录
    print("中文语料分词结束")


def corpus2Bunch(wordbag_path, seg_path):
    catelist = os.listdir(seg_path)  # 获取seg_path下的所有子目录，也就是分类信息
    # 创建一个Bunch实例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)
    '''
    extend(addlist)是python list中的函数，意思是用新的list（addlist）去扩充
    原来的list
    '''
    # 获取每个目录下所有的文件
    for mydir in catelist:
        class_path = seg_path + mydir + "/"  # 拼出分类子目录的路径
        file_list = os.listdir(class_path)  # 获取class_path下的所有文件
        for file_path in file_list:  # 遍历类别目录下文件
            fullname = class_path + file_path  # 拼出文件名全路径
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(readfile(fullname))  # 读取文件内容
            '''append(element)是python list中的函数，意思是向原来的list中添加element，注意与extend()函数的区别'''
    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("构建文本对象结束")


path1 = sys.path[0]
path1 = path1.replace("\\base_library.zip", "")
print(path1)


def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path=None):
    stpwrdlst = readfile(stopword_path).splitlines()
    bunch = readbunchobj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})

    if train_tfidf_path is not None:
        trainbunch = readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.6)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_
        weight = tfidfspace.tdm.toarray()  # 这是矩阵
        word = vectorizer.get_feature_names()  # 这是列表
        '''
        a = []
        for key, value in tfidfspace.vocabulary.items():
            a.append(key + "\t" + str(value))
        savefile("vocabulary" + ".txt", '\r\n'.join(a).encode('utf-8'))
        '''
        n = len(word)  # 词个数
        m = len(weight)  # 文本个数
        list2 = []
        for i in tfidfspace.label:
            address_index = [x for x in range(m) if tfidfspace.label[x] == i]
            list2.append([i, address_index])
        dict_address = dict(list2)
        text3 = ""
        for i, j in dict_address.items():
            text3 = text3 + "," + i
        text3 = text3 + "\r\n"
        z = 100
        for i in range(n):  # 词典评分系统
            text3 = text3 + word[i]
            for j, k in dict_address.items():
                c = []
                for l in k:
                    c.append(weight[l][i])
                p = max(c) * z
                text3 = text3 + ',' + str(round(p, 1))
            text3 = text3 + "\r\n"
        savefile(path1 + "/vocabulary.csv", text3.encode('GBK'))
    writebunchobj(space_path, tfidfspace)
    print("if-idf词向量空间实例创建成功")


# 检测一级二级路径文件夹中是否有非txt文件或文件夹
def test_file(path):
    ls = os.listdir(path)
    if ls == []:
        print(path + "文件夹为空，请检测训练数据文件夹内容")
        sys.exit(-1)
    else:
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                lss = os.listdir(c_path)
                for j in lss:
                    d_path = os.path.join(c_path, j)
                    if not os.path.isdir(d_path):
                        if len(j) < 5:
                            print(d_path + "不是txt文件，请检查训练数据文件夹内容")
                            sys.exit(-1)
                        else:
                            if d_path[-4:] != ".txt":
                                print(d_path + "不是txt文件，请检查训练数据文件夹内容")
                                sys.exit(-1)
                    else:
                        print(d_path + "不是txt文件，而是文件夹，请检查训练数据内容")
                        sys.exit(-1)
            else:
                print(c_path + "不是文件夹，请检查训练数据文件夹内容")
                sys.exit(-1)


def custom_rmdir(path):
    # 使用递归函数，一次删除文件夹里面的文件，并删除文件夹
    if len(os.listdir(path)):
        for sub_name in os.listdir(path):
            sub_path = os.path.join(path, sub_name)
            if os.path.isfile(sub_path):
                os.remove(sub_path)
            else:
                custom_rmdir(sub_path)
    os.rmdir(path)


path1 = sys.path[0]
path1 = path1.replace("\\base_library.zip", "")
print(path1)

print('此程序运行时，请勿打开exe文件夹内"训练数据分词后"文件夹，否则会出错')
corpus_path = path1 + "/训练数据/"  # 未分词分类语料库路径
seg_path = path1 + "/训练数据分词后/"  # 分词后分类语料库路径
print("正在检测训练数据文件夹是否干净")
if not os.path.exists(corpus_path):
    os.makedirs(corpus_path)
test_file(corpus_path)
print("训练数据文件夹检测通过")

print("正在清空训练数据分词后文件夹")
if not os.path.exists(seg_path):
    os.makedirs(seg_path)
list1 = os.listdir(seg_path)
for i in list1:
    c_path = os.path.join(seg_path, i)
    if os.path.isdir(c_path):  # 是文件夹
        lss = os.listdir(c_path)
        for j in lss:
            d_path = os.path.join(c_path, j)
            if not os.path.isdir(d_path):  # 不是文件夹
                os.remove(d_path)
            else:  # 是文件夹
                custom_rmdir(d_path)
    else:  # 不是文件夹
        os.remove(c_path)
print("训练数据分词后文件夹已清空")

# 对训练集进行分词
corpus_segment(corpus_path, seg_path)

# 对训练集进行Bunch化操作：
if not os.path.exists(path1 + "/train_word_bag"):
    os.makedirs(path1 + "/train_word_bag")
wordbag_path = path1 + "/train_word_bag/train_set.dat"  # Bunch存储路径
seg_path = path1 + "/训练数据分词后/"  # 分词后分类语料库路径
corpus2Bunch(wordbag_path, seg_path)

stopword_path = path1 + "/train_word_bag/hlt_stop_words.txt"
if not os.path.exists(stopword_path):
    print(stopword_path + "文件不存在，请检测train_word_bag文件夹完整性")
    sys.exit(-1)
bunch_path = path1 + "/train_word_bag/train_set.dat"
space_path = path1 + "/train_word_bag/tfidfspace.dat"
vector_space(stopword_path, bunch_path, space_path)
