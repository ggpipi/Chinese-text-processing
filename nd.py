#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import os
import jieba
import docx2txt
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import shutil
# from sklearn import metrics  #预测精度用 在分类系统中不需要
import pickle
import datetime
import re
import importlib
import chardet  # 转码识别

importlib.reload(sys)
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
from win32com import client as wc


# 保存至文件
def savefile(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)


# 读取文件
def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


# 写入bunch对象
def writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


# 读取bunch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


# 上面复制的
# 重写一个客户专用分词函数
def corpus_segment2(corpus_path, seg_path):
    print("正在分词，请稍候")
    if not os.path.exists(seg_path):  # 是否存在分词目录，如果没有则创建该目录
        os.makedirs(seg_path)

    file_list = os.listdir(corpus_path)  # 获取未分词语料库中某一类别中的所有文本
    '''
    train_corpus/中的
    21.txt,
    22.txt,
    23.txt
    ...
    file_list=['21.txt','22.txt',...]
    '''
    for file_path in file_list:  # 遍历类别目录下的所有文件
        fullname = corpus_path + file_path  # 拼出文件名全路径如：train_corpus/21.txt
        content = readfile(fullname)  # 读取文件内容
        '''此时，content里面存贮的是原文本的所有字符，例如多余的空格、空行、回车等等，
        接下来，我们需要把这些无关痛痒的字符统统去掉，变成只有标点符号做间隔的紧凑的文本内容
        '''
        content1 = content.decode('utf-8')
        content1 = re.sub("[^\u4e00-\u9fa5]", "", content1)  # unicode中文范围
        content = content1.encode('utf-8')
        content_seg = jieba.cut(content)  # 为文件内容分词
        savefile(seg_path + file_path, ' '.join(content_seg).encode('utf-8'))  # 将处理后的文件保存到分词后语料目录
    print("中文语料分词结束")


# 客户专用Bunch
def corpus2Bunch2(wordbag_path, seg_path):
    # 创建一个Bunch实例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    '''
    extend(addlist)是python list中的函数，意思是用新的list（addlist）去扩充
    原来的list
    '''
    # 获取每个目录下所有的文件
    class_path = seg_path  # 分类子目录的路径
    file_list = os.listdir(class_path)  # 获取class_path下的所有文件
    bunch.target_name.extend(file_list)
    for file_path in file_list:  # 遍历类别目录下文件
        fullname = class_path + file_path  # 拼出文件名全路径
        bunch.label.append("Don't know")
        bunch.filenames.append(fullname)
        bunch.contents.append(readfile(fullname))  # 读取文件内容
        '''append(element)是python list中的函数，意思是向原来的list中添加element，注意与extend()函数的区别'''
    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("构建文本对象结束")


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
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_

    writebunchobj(space_path, tfidfspace)
    print("if-idf词向量空间实例创建成功")


def getTxtFromWord(a_path, b_path):  # 批量转入a文件夹内docx成txt到b文件夹
    file_list = os.listdir(a_path)
    for file_path in file_list:  # 遍历类别目录下的所有文件
        fullname = a_path + file_path  # 拼出文件名全路径如：train_corpus/art/21.txt
        newname = file_path.replace('docx', 'txt')
        newfullname = b_path + newname
        text = docx2txt.process(fullname)  # 读取文件内容
        savefile(newfullname, text.encode('utf-8'))


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


# 递归删除文件夹内所有文件，保留文件夹
def del_file(path):
    ls = os.listdir(path)
    for ii in ls:
        c_path = os.path.join(path, ii)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


# 删除文件夹内所有文件和文件夹，保留根文件夹
def del_file1(path):
    ls = os.listdir(path)
    for ii in ls:
        c_path = os.path.join(path, ii)
        if os.path.isdir(c_path):
            custom_rmdir(c_path)
        else:
            os.remove(c_path)


# 整理文件夹
def clean_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        del_file1(path)


print("运行时，请不要打开“model”文件夹，以免出错！")
# 打印看看命令行参数是什么
for i in range(len(sys.argv)):
    print("参数", i, sys.argv[i])

path1 = sys.path[0]
path1 = path1.replace("\\base_library.zip", "")
print(path1)

corpus_path = sys.argv[1]
corpus_path1 = path1 + "/model/测试数据中间站1/"

# 一点点清洁工作
clean_dir(corpus_path1)
clean_dir(path1 + "/model/测试数据分词后")
clean_dir(path1 + "/model/test_word_bag")

word = wc.Dispatch("Word.Application")  # word应用
word.Visible = 0
word.DisplayAlerts = 0
if not os.path.exists(corpus_path1):
    os.makedirs(corpus_path1)
print("正在识别并整理目标文件夹下文件...")
for file in os.listdir(corpus_path):  # 识别整理目标文件夹下所有文件
    a = os.path.splitext(file)
    full_name = os.path.join(corpus_path, file)
    full_name1 = os.path.join(corpus_path1, file)
    if a[1] == '.txt':
        testtxt = readfile(full_name)
        if chardet.detect(testtxt)['encoding'] != 'utf-8':
            codename = chardet.detect(testtxt)['encoding']
            testtxt = testtxt.decode(codename)
            savefile(full_name1, testtxt.encode('utf-8'))
        else:
            shutil.copy(full_name, corpus_path1)
    else:
        if a[1] == '.docx':
            textdocx = docx2txt.process(full_name)
            savefile(full_name1, textdocx.encode('utf-8'))
        else:
            if a[1] == '.pdf':
                fp = open(full_name, 'rb')
                # 来创建一个pdf文档分析器
                praser = PDFParser(fp)
                # 创建一个PDF文档
                doc = PDFDocument()
                # 连接分析器 与文档对象
                praser.set_document(doc)
                doc.set_parser(praser)

                # 提供初始化密码
                # 如果没有密码 就创建一个空的字符串
                doc.initialize()
                # 检测文档是否提供txt转换，不提供就忽略
                if not doc.is_extractable:
                    raise PDFTextExtractionNotAllowed
                else:
                    # 创建PDf 资源管理器 来管理共享资源
                    rsrcmgr = PDFResourceManager()
                    # 创建一个PDF设备对象
                    laparams = LAParams()
                    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
                    # 创建一个PDF解释器对象
                    interpreter = PDFPageInterpreter(rsrcmgr, device)

                    # 循环遍历列表，每次处理一个page的内容
                    for page in doc.get_pages():  # doc.get_pages() 获取page列表
                        interpreter.process_page(page)
                        # 接受该页面的LTPage对象
                        layout = device.get_result()
                        # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象
                        # 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等
                        # 想要获取文本就获得对象的text属性，
                        for x in layout:
                            if (isinstance(x, LTTextBoxHorizontal)):
                                with open(full_name1, 'ab') as f:
                                    results = x.get_text()
                                    f.write((results + '\r\n').encode('utf-8'))
            else:
                if a[1] == '.doc':
                    doc = word.Documents.Open(full_name)
                    doc.SaveAs(FileFormat=7, FileName=full_name1)
                    doc.Close()
                    textdoc = readfile(full_name1)
                    textdoc = textdoc.decode("GBK")
                    savefile(full_name1, textdoc.encode("utf-8"))
                else:
                    print(corpus_path + "存在无法识别的文件，请检查文件夹内容")
                    sys.exit(-1)
word.Quit()  # 退出word

# 对测试集进行分词
# 若采用命令行传入参数方式，改下面
# corpus_path = sys.argv[2]  # 未分词分类语料库路径（绝对路径）
# corpus_path = "./测试数据/"  # 未分词分类语料库路径（相对路径）
seg_path = path1 + "/model/测试数据分词后/"  # 分词后分类语料库路径
corpus_segment2(corpus_path1, seg_path)

# 对测试集进行Bunch化操作：
wordbag_path = path1 + "/model/test_word_bag/test_set.dat"  # Bunch存储路径
seg_path = path1 + "/model/测试数据分词后/"  # 分词后分类语料库路径
corpus2Bunch2(wordbag_path, seg_path)

stopword_path = path1 + "/train_word_bag/hlt_stop_words.txt"
bunch_path = path1 + "/model/test_word_bag/test_set.dat"
space_path = path1 + "/model/test_word_bag/testspace.dat"
train_tfidf_path = path1 + "/train_word_bag/tfidfspace.dat"
vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)

# 导入训练集
trainpath = path1 + "/train_word_bag/tfidfspace.dat"
train_set = readbunchobj(trainpath)

# 导入测试集
testpath = path1 + "/model/test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)

# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)

# 预测分类结果
predicted = clf.predict(test_set.tdm)
predicted2 = clf.predict_proba(test_set.tdm)
list1 = []

# 注释的是原版
# for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
for file_name, expct_cate in zip(os.listdir(corpus_path1), predicted):
    # text1 = file_name + ": 实际类别:" + flabel + " -->预测类别:" + expct_cate
    text1 = corpus_path + file_name + "\t" + expct_cate
    list1.append(text1)
    print(text1)

text2 = ""
for i in train_set.target_name:
    text2 = text2 + "," + i

for file_name, i in zip(os.listdir(corpus_path1), predicted2):
    text2 = text2 + "\r\n"
    text2 = text2 + corpus_path + file_name
    for j in i:
        text2 = text2 + ',' + str(round(j, 2))

a = path1 + "/输出结果储存"
if not os.path.exists(a):
    os.makedirs(a)
name = a + '/结果' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
savefile(name + ".csv", text2.encode('GBK'))  # csv默认是gbk编码
savefile(name + ".txt", '\r\n'.join(list1).encode('utf-8'))

print("预测完毕")
print(name + ".csv")
print(name + ".txt")

"""
# 计算分类精度：
因为不知道是啥，所以算精度没意义了
def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))


metrics_result(test_set.label, predicted)
"""
