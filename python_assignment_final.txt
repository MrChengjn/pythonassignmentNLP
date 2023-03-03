#this file is for the 2021 python class' hard assignment
#editor:cjn,2012628
#the last modified date :2021/12/4

#文件框架：
#一、获取语料
#   1.1 读取csv文件
#   1.2 使用爬虫爬取文件中url地址对应的文章内容
#   1.3 对爬取内容可用性筛选，爬取失败的文件删除，得到可用文件索引
#二、语料清洗
#   2.1 分句，去标点，分词清洗，不去停用词，为word2vec词汇向量化作准备
#   2.2 去停用词，为随机森林训练做准备
#三、特征工程
#   3.1 使用word2vec模型将词向量化
#   3.2 读取word2vec模型，获取每一份文件的平均向量
#四、模型拟合
#   4.1 使用随机森林模型拟合
#五、评价指标
#   5.1 输出混淆矩阵和训练报告
#参考资料：https://cloud.tencent.com/developer/article/1348236
#https://blog.csdn.net/FRBeVrQbN4L/article/details/109698919
#https://cloud.tencent.com/developer/article/1059336
#https://github.com/duoergun0729/nlp

#使用python及相应库版本
#python=3.8.8
#beautifulsoup4=4.9.3
#zhon=1.1.5
#jieba=0.42.1
#gensim=4.0.1
#scikit-learn=0.24.1

#使用帮助请到文档最后阅读帮助文档（第339行）

import urllib.request
import random
import os
from csv import reader
from bs4 import BeautifulSoup
import re
import zhon
import jieba.posseg as pseg
from zhon.hanzi import punctuation
import multiprocessing
from gensim.models import Word2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#使用随机头防止被网站识别为脚本获取信息失败
ua_list = [
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11',
    'User-Agent:Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11',
    'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0',
    ' Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1',
    ' Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
]

#一、获取语料
#1.1 读取csv文件
#读取训练集
path_train = os.getcwd()+'\\WeFEND-AAAI20-master\\data\\train\\news.csv'
path_train = path_train.replace('\\','/')
with open(path_train,'r',encoding='UTF-8') as csv_file:
    csv_reader_train = reader(csv_file)
    list_of_news_train = list(csv_reader_train)
#读取测试集
path_test = os.getcwd()+'\\WeFEND-AAAI20-master\\data\\test\\news.csv'
path_test = path_test.replace('\\','/')
with open(path_test,'r',encoding='UTF-8') as csv_file:
    csv_reader_test = reader(csv_file)
    list_of_news_test = list(csv_reader_test)
#读取无标签数据
path_unlabeled = os.getcwd()+'\\WeFEND-AAAI20-master\\data\\unlabeled data\\news.csv'
path_unlabeled = path_unlabeled.replace('\\','/')
with open(path_unlabeled,'r',encoding='UTF-8') as csv_file:
    csv_reader_unlabeled = reader(csv_file)
    list_of_news_unlabeled = list(csv_reader_unlabeled)

#1.2 使用爬虫爬取文件中url地址对应的文章内容
#生成一个对url地址的访问
def request_html(url):
    headers={'User-Agent':random.choice(ua_list)}
    request = urllib.request.Request(url, headers=headers)
    return request
#爬取的内容写入txt文件
def parse_html(html, f):
    soup = BeautifulSoup(html, 'lxml')
    line_name = soup.select('.rich_media_content')
    title_name = soup.select('.rich_media_title')
    if(len(title_name)!=0):
        for item in title_name:
            title = item.text
            f.write(title)
    if(len(line_name)!=0):
        for item in line_name:
            content = item.text
            f.write('\n'+content)

#对整个csv文件遍历，爬取全部网址
#创建文件夹保存结果
path_now = os.getcwd()
path_now = path_now.replace('\\','/')
os.mkdir(path_now+'/'+'essays')
path_essays = path_now+'/'+'essays'
#爬取训练集
os.mkdir(path_essays+'/'+'train')
for i in range(1,len(list_of_news_train)):
    f = open(path_essays+'/train/news_'+str(i)+'.txt', 'w', encoding='utf8')
    url = list_of_news_train[i][2]
    request = request_html(url)
    html = urllib.request.urlopen(request).read().decode('utf8')
    parse_html(html,f)
    f.close()
#爬取测试集
os.mkdir(path_essays+'/'+'test')
for i in range(1,len(list_of_news_test)):
    f = open(path_essays+'/test/news_'+str(i)+'.txt', 'w', encoding='utf8')
    url = list_of_news_test[i][2]
    request = request_html(url)
    html = urllib.request.urlopen(request).read().decode('utf8')
    parse_html(html,f)
    f.close()
#爬取无标签数据
os.mkdir(path_essays+'/'+'unlabeled')
for i in range(1,len(list_of_news_unlabeled)):
    f = open(path_essays+'/unlabeled/news_'+str(i)+'.txt', 'w', encoding='utf8')
    url = list_of_news_unlabeled[i][1]
    request = request_html(url)
    html = urllib.request.urlopen(request).read().decode('utf8')
    parse_html(html,f)
    f.close()

#1.3 对爬取内容可用性筛选，爬取失败的文件删除，得到可用文件索引
#筛选训练集
right_index_train = []
for i in range(1,len(list_of_news_train)):
    if os.path.exists(path_essays+'/train/news_'+str(i)+'.txt'):
        if not os.path.getsize(path_essays+'/train/news_'+str(i)+'.txt'):
            os.remove(path_essays+'/train/news_'+str(i)+'.txt')
        else:
            right_index_train.append(i)
#筛选测试集
right_index_test = []
for i in range(1,len(list_of_news_test)):
    if os.path.exists(path_essays+'/test/news_'+str(i)+'.txt'):
        if not os.path.getsize(path_essays+'/test/news_'+str(i)+'.txt'):
            os.remove(path_essays+'/test/news_'+str(i)+'.txt')
        else:
            right_index_test.append(i)
#筛选无标签数据
right_index_unlabeled = []
for i in range(1,len(list_of_news_unlabeled)):
    if os.path.exists(path_essays+'/unlabeled/news_'+str(i)+'.txt'):
        if not os.path.getsize(path_essays+'/unlabeled/news_'+str(i)+'.txt'):
            os.remove(path_essays+'/unlabeled/news_'+str(i)+'.txt')
        else:
            right_index_unlabeled.append(i)

#二、语料清洗
#2.1 分句，去标点，分词清洗，不去停用词，为word2vec词汇向量化作准备
#获取训练集语料
text_chinese_train=''
for i in range(len(right_index_train)):
    file_object=open(path_essays+'/train/news_'+str(right_index_train[i])+'.txt','r', encoding='utf-8')
    all_the_text=file_object.read()
    file_object.close()
    text_chinese_train = text_chinese_train + all_the_text
#获取测试集语料
text_chinese_test=''
for i in range(len(right_index_test)):
    file_object=open(path_essays+'/test/news_'+str(right_index_test[i])+'.txt','r', encoding='utf-8')
    all_the_text=file_object.read()
    file_object.close()
    text_chinese_test = text_chinese_test + all_the_text
#获取无标签数据语料
text_chinese_unlabeled=''
for i in range(len(right_index_unlabeled)):
    file_object=open(path_essays+'/unlabeled/news_'+str(right_index_unlabeled[i])+'.txt','r', encoding='utf-8')
    all_the_text=file_object.read()
    file_object.close()
    text_chinese_unlabeled = text_chinese_unlabeled + all_the_text
#获取总语料库
text_chinese_total = text_chinese_train + text_chinese_test + text_chinese_unlabeled
#分句
rst_total = re.findall(zhon.hanzi.sentence,text_chinese_total)
#对每一句分词
for i in range(len(rst_total)):
    #去标点符号
    chi_nopuc = re.sub("[{}]+".format(punctuation), "", rst_total[i])
    #分词
    words = pseg.cut(chi_nopuc)
    final = []
    for chi in words:
        final.append(chi)
    rst_total[i] = final
#获得纯文本类型的结果（原来为pair，可以输出flag标志）
rstword_total = [[] for i in range(len(rst_total))]
for i in range(len(rst_total)):
    for j in range(len(rst_total[i])):
        rstword_total[i].append(rst_total[i][j].word)
        
#2.2 去停用词，为随机森林训练做准备
#清洗训练集
clean_reviews_train = []
labels_train = []
for i in range(len(right_index_train)):
    file_object=open(path_essays+'/train/news_'+str(right_index_train[i])+'.txt','r', encoding='utf-8')
    all_the_text=file_object.read()
    file_object.close()
    #分句
    rst_train = re.findall(zhon.hanzi.sentence,all_the_text)
    for j in range(len(rst_train)):
        #去标点符号
        chi_nopuc = re.sub("[{}]+".format(punctuation), "", rst_train[j])
        #分词
        words = pseg.cut(chi_nopuc)
        #去停用词
        f = open("中文停用词.txt",'r',encoding = 'UTF-8')
        stopwords_n = f.readlines()
        f.close()
        stopwords = [sw.strip().replace('\n','') for sw in stopwords_n]
        final = []
        for chi in words:
            if chi.word not in stopwords:
                final.append(chi)
        #输出
        rst_train[j] = final
    #获得纯文本类型的结果（原来为pair，可以输出flag标志）
    rstword_train = [[] for j in range(len(rst_train))]
    for j in range(len(rst_train)):
        for k in range(len(rst_train[j])):
            rstword_train[j].append(rst_train[j][k].word)
    clean_reviews_train.append(rstword_train)
    labels_train.append(list_of_news_train[right_index_train[i]][5])
#清洗测试集
clean_reviews_test = []
labels_test = []
for i in range(len(right_index_test)):
    file_object=open(path_essays+'/test/news_'+str(right_index_test[i])+'.txt','r', encoding='utf-8')
    all_the_text=file_object.read()
    file_object.close()
    #分句
    rst_test = re.findall(zhon.hanzi.sentence,all_the_text)
    for j in range(len(rst_test)):
        #去标点符号
        chi_nopuc = re.sub("[{}]+".format(punctuation), "", rst_test[j])
        #分词
        words = pseg.cut(chi_nopuc)
        #去停用词
        f = open("中文停用词.txt",'r',encoding = 'UTF-8')
        stopwords_n = f.readlines()
        f.close()
        stopwords = [sw.strip().replace('\n','') for sw in stopwords_n]
        final = []
        for chi in words:
            if chi.word not in stopwords:
                final.append(chi)
        #输出
        rst_test[j] = final
    #获得纯文本类型的结果（原来为pair，可以输出flag标志）
    rstword_test = [[] for j in range(len(rst_test))]
    for j in range(len(rst_test)):
        for k in range(len(rst_test[j])):
            rstword_test[j].append(rst_test[j][k].word)
    clean_reviews_test.append(rstword_test)
    labels_test.append(list_of_news_test[right_index_test[i]][5])

#三、特征工程
#3.1 使用word2vec模型将词向量化
model = Word2Vec(rstword_total,vector_size=500,window=5,min_count=5,workers=multiprocessing.cpu_count())
model.save('news_word2vec_500.w2v')
model.wv.save_word2vec_format('news_word2vec_500.bin',binary=False)

#3.2 读取word2vec模型，获取每一份文件的平均向量
#处理文章的函数
def makeFeatureVec(words,model,num_features):
    #初始化0向量组
    featureVec = np.zeros((num_features),dtype='float32')
    #初始化单词数
    nwords = 1 #初始化为1防止除法失败
    #返回字典
    index2word_set = model.wv.index_to_key
    #检查单词是否在字典中
    for word in words:
        for i in range(len(word)):
            #如果在
            if word[i] in index2word_set:
                #单词数加1，向量相加
                nwords+=1
                featureVec = np.add(featureVec,model.wv[word[i]])
    #返回平均向量
    return np.divide(featureVec,nwords)
#处理集的函数
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype='float32')
    #遍历集合中所有的文章
    for review in reviews:
        if counter % 1000 == 0:
            print('Review %d of %d' % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
        counter += 1
    return reviewFeatureVecs
#获取训练集的向量
trainDataVecs_train = getAvgFeatureVecs(clean_reviews_train, model, 500)
#获取测试集的向量
trainDataVecs_test = getAvgFeatureVecs(clean_reviews_test, model, 500)

#四、模型拟合
#4.1 使用随机森林模型拟合
#初始化100棵RF分类器
forest = RandomForestClassifier(n_estimators=100)
#设置训练集测试集
X_train = trainDataVecs_train
Y_train = labels_train
X_test = trainDataVecs_test
Y_test = labels_test
#开始训练
forest = forest.fit(X_train, Y_train)

#五、评价指标
#5.1 输出混淆矩阵和训练报告
#预测
Y_pred = forest.predict(X_test)
#输出混淆矩阵和报告
conf_matrix = confusion_matrix(Y_test,Y_pred)
class_report = classification_report(Y_test, Y_pred)
print(conf_matrix)
print(class_report)

'''
附件1：帮助文档
使用该python程序时，直接将该程序、停用词文件和数据集文件夹放在同一根目录下即可
架构示意：根目录-WeFEND-AAAI20-master（文件夹，下载的文件直接解压）
               -python_assignment_final.py（该python程序）
               -中文停用词.txt（所用停用词文件）
运行该程序。成功的输出结果应该与下段文字相类似：
（建议运行方式：使用anaconda中的spyder运行，其他平台暂未尝试）


Review 0 of 3262
Review 1000 of 3262
Review 2000 of 3262
Review 3000 of 3262
Review 0 of 4131
Review 1000 of 4131
Review 2000 of 4131
Review 3000 of 4131
Review 4000 of 4131
[[3825   17]
 [ 170  119]]
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      3842
           1       0.88      0.41      0.56       289

    accuracy                           0.95      4131
   macro avg       0.92      0.70      0.77      4131
weighted avg       0.95      0.95      0.95      4131


以Review开头的部分是在计算文章向量化时输出，每1000个文章为1轮
如果遇到异常终止，方便定位异常文件
第一次计算的是训练集，第二次计算的是测试集
接下来依次输出的是测试集的混淆矩阵和本次拟合的数据报告

注意：输出时可能还会输出以下段落：


Building prefix dict from the default dictionary ...
Loading model from cache *:\****\jieba.cache（为一个文件路径）
Loading model cost * seconds.（为一个数字）
Prefix dict has been built successfully.


为正常情况，这并不影响我们程序的功能。
程序运行成功之后根目录应该有如下结构：
架构示意：根目录-WeFEND-AAAI20-master（文件夹）
               -python_assignment_final.py（该python程序）
               -中文停用词.txt（所用停用词文件）
               -essays（文件夹）
               -news_word2vec_500.bin（bin文件）
               -news_word2vec_500.w2v（w2v文件）
               -news_word2vec_500.w2v.syn1neg.npy（npy文件）
               -news_word2vec_500.w2v.wv.vectors.npy（npy文件）

其中，essays文件夹中包含三个名为train、test、unlabeled文件夹，
分别存储训练集、测试集和无标签集的爬虫数据（里面是以序号编码的形如
news_1.txt的txt文件）
下面的一个bin文件、一个w2v文件和两个npy文件都是存储word2vec训练模型的文件

debug的一些帮助：
1.如果在爬虫过程中出现类似
[WinError]一个请求已被阻止
的报错，是由于网络不稳定导致的。更换一个稳定的网络即可解决
2.如果遇到“某一个库或者数据没有某某参数”等错误，很可能是电脑上的包
版本不对导致的。尝试卸载电脑上当前安装的所用库下载文档开头的版本
3.其他的一些越界无效数据等报错，请检查数据输入

参考运行时间：
10小时左右。其中，爬虫时间（30000+个网页）约8-9小时，
数据向量化和训练模型约0.5-1小时
时间仅供参考，这与你的网速和电脑性能有关
'''


'''
附件2：中文停用词
使用该程序时，请将下面的内容复制保存为“中文停用词.txt”文件放在指定目录下


啊 
阿 
哎 
哎呀 
哎哟 
唉 
俺 
俺们 
按 
按照 
吧 
吧哒 
把 
罢了 
被 
本 
本着 
比 
比方 
比如 
鄙人 
彼 
彼此 
边 
别 
别的 
别说 
并 
并且 
不比 
不成 
不单 
不但 
不独 
不管 
不光 
不过 
不仅 
不拘 
不论 
不怕 
不然 
不如 
不特 
不惟 
不问 
不只 
朝 
朝着 
趁 
趁着 
乘 
冲 
除 
除此之外 
除非 
除了 
此 
此间 
此外 
从 
从而 
打 
待 
但 
但是 
当 
当着 
到 
得 
的 
的话 
等 
等等 
地 
第 
叮咚 
对 
对于 
多 
多少 
而 
而况 
而且 
而是 
而外 
而言 
而已 
尔后 
反过来 
反过来说 
反之 
非但 
非徒 
否则 
嘎 
嘎登 
该 
赶 
个 
各 
各个 
各位 
各种 
各自 
给 
根据 
跟 
故 
故此 
固然 
关于 
管 
归 
果然 
果真 
过 
哈 
哈哈 
呵 
和 
何 
何处 
何况 
何时 
嘿 
哼 
哼唷 
呼哧 
乎 
哗 
还是 
还有 
换句话说 
换言之 
或 
或是 
或者 
极了 
及 
及其 
及至 
即 
即便 
即或 
即令 
即若 
即使 
几 
几时 
己 
既 
既然 
既是 
继而 
加之 
假如 
假若 
假使 
鉴于 
将 
较 
较之 
叫 
接着 
结果 
借 
紧接着 
进而 
尽 
尽管 
经 
经过 
就 
就是 
就是说 
据 
具体地说 
具体说来 
开始 
开外 
靠 
咳 
可 
可见 
可是 
可以 
况且 
啦 
来 
来着 
离 
例如 
哩 
连 
连同 
两者 
了 
临 
另 
另外 
另一方面 
论 
嘛 
吗 
慢说 
漫说 
冒 
么 
每 
每当 
们 
莫若 
某 
某个 
某些 
拿 
哪 
哪边 
哪儿 
哪个 
哪里 
哪年 
哪怕 
哪天 
哪些 
哪样 
那 
那边 
那儿 
那个 
那会儿 
那里 
那么 
那么些 
那么样 
那时 
那些 
那样 
乃 
乃至 
呢 
能 
你 
你们 
您 
宁 
宁可 
宁肯 
宁愿 
哦 
呕 
啪达 
旁人 
呸 
凭 
凭借 
其 
其次 
其二 
其他 
其它 
其一 
其余 
其中 
起 
起见 
岂但 
恰恰相反 
前后 
前者 
且 
然而 
然后 
然则 
让 
人家 
任 
任何 
任凭 
如 
如此 
如果 
如何 
如其 
如若 
如上所述 
若 
若非 
若是 
啥 
上下 
尚且 
设若 
设使 
甚而 
甚么 
甚至 
省得 
时候 
什么 
什么样 
使得 
是 
是的 
首先 
谁 
谁知 
顺 
顺着 
似的 
虽 
虽然 
虽说 
虽则 
随 
随着 
所 
所以 
他 
他们 
他人 
它 
它们 
她 
她们 
倘 
倘或 
倘然 
倘若 
倘使 
腾 
替 
通过 
同 
同时 
哇 
万一 
往 
望 
为 
为何 
为了 
为什么 
为着 
喂 
嗡嗡 
我 
我们 
呜 
呜呼 
乌乎 
无论 
无宁 
毋宁 
嘻 
吓 
相对而言 
像 
向 
向着 
嘘 
呀 
焉 
沿 
沿着 
要 
要不 
要不然 
要不是 
要么 
要是 
也 
也罢 
也好 
一 
一般 
一旦 
一方面 
一来 
一切 
一样 
一则 
依 
依照 
矣 
以 
以便 
以及 
以免 
以至 
以至于 
以致 
抑或 
因 
因此 
因而 
因为 
哟 
用 
由 
由此可见 
由于 
有 
有的 
有关 
有些 
又 
于 
于是 
于是乎 
与 
与此同时 
与否 
与其 
越是 
云云 
哉 
再说 
再者 
在 
在下 
咱 
咱们 
则 
怎 
怎么 
怎么办 
怎么样 
怎样 
咋 
照 
照着 
者 
这 
这边 
这儿 
这个 
这会儿 
这就是说 
这里 
这么 
这么点儿 
这么些 
这么样 
这时 
这些 
这样 
正如 
吱 
之 
之类 
之所以 
之一 
只是 
只限 
只要 
只有 
至 
至于 
诸位 
着 
着呢 
自 
自从 
自个儿 
自各儿 
自己 
自家 
自身 
综上所述 
总的来看 
总的来说 
总的说来 
总而言之 
总之 
纵 
纵令 
纵然 
纵使 
遵照 
作为 
兮 
呃 
呗 
咚 
咦 
喏 
啐 
喔唷 
嗬 
嗯 
嗳 
啊哈 
啊呀 
啊哟 
挨次 
挨个 
挨家挨户 
挨门挨户 
挨门逐户 
挨着 
按理 
按期 
按时 
按说 
暗地里 
暗中 
暗自 
昂然 
八成 
白白 
半 
梆 
保管 
保险 
饱 
背地里 
背靠背 
倍感 
倍加 
本人 
本身 
甭 
比起 
比如说 
比照 
毕竟 
必 
必定 
必将 
必须 
便 
别人 
并非 
并肩 
并没 
并没有 
并排 
并无 
勃然 
不 
不必 
不常 
不大 
不得 
不得不 
不得了 
不得已 
不迭 
不定 
不对 
不妨 
不管怎样 
不会 
不仅仅 
不仅仅是 
不经意 
不可开交 
不可抗拒 
不力 
不了 
不料 
不满 
不免 
不能不 
不起 
不巧 
不然的话 
不日 
不少 
不胜 
不时 
不是 
不同 
不能 
不要 
不外 
不外乎 
不下 
不限 
不消 
不已 
不亦乐乎 
不由得 
不再 
不择手段 
不怎么 
不曾 
不知不觉 
不止 
不止一次 
不至于 
才 
才能 
策略地 
差不多 
差一点 
常 
常常 
常言道 
常言说 
常言说得好 
长此下去 
长话短说 
长期以来 
长线 
敞开儿 
彻夜 
陈年 
趁便 
趁机 
趁热 
趁势 
趁早 
成年 
成年累月 
成心 
乘机 
乘胜 
乘势 
乘隙 
乘虚 
诚然 
迟早 
充分 
充其极 
充其量 
抽冷子 
臭 
初 
出 
出来 
出去 
除此 
除此而外 
除此以外 
除开 
除去 
除却 
除外 
处处 
川流不息 
传 
传说 
传闻 
串行 
纯 
纯粹 
此后 
此中 
次第 
匆匆 
从不 
从此 
从此以后 
从古到今 
从古至今 
从今以后 
从宽 
从来 
从轻 
从速 
从头 
从未 
从无到有 
从小 
从新 
从严 
从优 
从早到晚 
从中 
从重 
凑巧 
粗 
存心 
达旦 
打从 
打开天窗说亮话 
大 
大不了 
大大 
大抵 
大都 
大多 
大凡 
大概 
大家 
大举 
大略 
大面儿上 
大事 
大体 
大体上 
大约 
大张旗鼓 
大致 
呆呆地 
带 
殆 
待到 
单 
单纯 
单单 
但愿 
弹指之间 
当场 
当儿 
当即 
当口儿 
当然 
当庭 
当头 
当下 
当真 
当中 
倒不如 
倒不如说 
倒是 
到处 
到底 
到了儿 
到目前为止 
到头 
到头来 
得起 
得天独厚 
的确 
等到 
叮当 
顶多 
定 
动不动 
动辄 
陡然 
都 
独 
独自 
断然 
顿时 
多次 
多多 
多多少少 
多多益善 
多亏 
多年来 
多年前 
而后 
而论 
而又 
尔等 
二话不说 
二话没说 
反倒 
反倒是 
反而 
反手 
反之亦然 
反之则 
方 
方才 
方能 
放量 
非常 
非得 
分期 
分期分批 
分头 
奋勇 
愤然 
风雨无阻 
逢 
弗 
甫 
嘎嘎 
该当 
概 
赶快 
赶早不赶晚 
敢 
敢情 
敢于 
刚 
刚才 
刚好 
刚巧 
高低 
格外 
隔日 
隔夜 
个人 
各式 
更 
更加 
更进一步 
更为 
公然 
共 
共总 
够瞧的 
姑且 
古来 
故而 
故意 
固 
怪 
怪不得 
惯常 
光 
光是 
归根到底 
归根结底 
过于 
毫不 
毫无 
毫无保留地 
毫无例外 
好在 
何必 
何尝 
何妨 
何苦 
何乐而不为 
何须 
何止 
很 
很多 
很少 
轰然 
后来 
呼啦 
忽地 
忽然 
互 
互相 
哗啦 
话说 
还 
恍然 
会 
豁然 
活 
伙同 
或多或少 
或许 
基本 
基本上 
基于 
极 
极大 
极度 
极端 
极力 
极其 
极为 
急匆匆 
即将 
即刻 
即是说 
几度 
几番 
几乎 
几经 
既…又 
继之 
加上 
加以 
间或 
简而言之 
简言之 
简直 
见 
将才 
将近 
将要 
交口 
较比 
较为 
接连不断 
接下来 
皆可 
截然 
截至 
藉以 
借此 
借以 
届时 
仅 
仅仅 
谨 
进来 
进去 
近 
近几年来 
近来 
近年来 
尽管如此 
尽可能 
尽快 
尽量 
尽然 
尽如人意 
尽心竭力 
尽心尽力 
尽早 
精光 
经常 
竟 
竟然 
究竟 
就此 
就地 
就算 
居然 
局外 
举凡 
据称 
据此 
据实 
据说 
据我所知 
据悉 
具体来说 
决不 
决非 
绝 
绝不 
绝顶 
绝对 
绝非 
均 
喀 
看 
看来 
看起来 
看上去 
看样子 
可好 
可能 
恐怕 
快 
快要 
来不及 
来得及 
来讲 
来看 
拦腰 
牢牢 
老 
老大 
老老实实 
老是 
累次 
累年 
理当 
理该 
理应 
历 
立 
立地 
立刻 
立马 
立时 
联袂 
连连 
连日 
连日来 
连声 
连袂 
临到 
另方面 
另行 
另一个 
路经 
屡 
屡次 
屡次三番 
屡屡 
缕缕 
率尔 
率然 
略 
略加 
略微 
略为 
论说 
马上 
蛮 
满 
没 
没有 
每逢 
每每 
每时每刻 
猛然 
猛然间 
莫 
莫不 
莫非 
莫如 
默默地 
默然 
呐 
那末 
奈 
难道 
难得 
难怪 
难说 
内 
年复一年 
凝神 
偶而 
偶尔 
怕 
砰 
碰巧 
譬如 
偏偏 
乒 
平素 
颇 
迫于 
扑通 
其后 
其实 
奇 
齐 
起初 
起来 
起首 
起头 
起先 
岂 
岂非 
岂止 
迄 
恰逢 
恰好 
恰恰 
恰巧 
恰如 
恰似 
千 
万 
千万 
千万千万 
切 
切不可 
切莫 
切切 
切勿 
窃 
亲口 
亲身 
亲手 
亲眼 
亲自 
顷 
顷刻 
顷刻间 
顷刻之间 
请勿 
穷年累月 
取道 
去 
权时 
全都 
全力 
全年 
全然 
全身心 
然 
人人 
仍 
仍旧 
仍然 
日复一日 
日见 
日渐 
日益 
日臻 
如常 
如此等等 
如次 
如今 
如期 
如前所述 
如上 
如下 
汝 
三番两次 
三番五次 
三天两头 
瑟瑟 
沙沙 
上 
上来 
上去 
一. 
一一 
一下 
一个 
一些 
一何 
一则通过 
一天 
一定 
一时 
一次 
一片 
一番 
一直 
一致 
一起 
一转眼 
一边 
一面 
上升 
上述 
上面 
下 
下列 
下去 
下来 
下面 
不一 
不久 
不变 
不可 
不够 
不尽 
不尽然 
不敢 
不断 
不若 
不足 
与其说 
专门 
且不说 
且说 
严格 
严重 
个别 
中小 
中间 
丰富 
为主 
为什麽 
为止 
为此 
主张 
主要 
举行 
乃至于 
之前 
之后 
之後 
也就是说 
也是 
了解 
争取 
二来 
云尔 
些 
亦 
产生 
人 
人们 
什麽 
今 
今后 
今天 
今年 
今後 
介于 
从事 
他是 
他的 
代替 
以上 
以下 
以为 
以前 
以后 
以外 
以後 
以故 
以期 
以来 
任务 
企图 
伟大 
似乎 
但凡 
何以 
余外 
你是 
你的 
使 
使用 
依据 
依靠 
便于 
促进 
保持 
做到 
傥然 
儿 
允许 
元／吨 
先不先 
先后 
先後 
先生 
全体 
全部 
全面 
共同 
具体 
具有 
兼之 
再 
再其次 
再则 
再有 
再次 
再者说 
决定 
准备 
凡 
凡是 
出于 
出现 
分别 
则甚 
别处 
别是 
别管 
前此 
前进 
前面 
加入 
加强 
十分 
即如 
却 
却不 
原来 
又及 
及时 
双方 
反应 
反映 
取得 
受到 
变成 
另悉 
只 
只当 
只怕 
只消 
叫做 
召开 
各人 
各地 
各级 
合理 
同一 
同样 
后 
后者 
后面 
向使 
周围 
呵呵 
咧 
唯有 
啷当 
喽 
嗡 
嘿嘿 
因了 
因着 
在于 
坚决 
坚持 
处在 
处理 
复杂 
多么 
多数 
大力 
大多数 
大批 
大量 
失去 
她是 
她的 
好 
好的 
好象 
如同 
如是 
始而 
存在 
孰料 
孰知 
它们的 
它是 
它的 
安全 
完全 
完成 
实现 
实际 
宣布 
容易 
密切 
对应 
对待 
对方 
对比 
小 
少数 
尔 
尔尔 
尤其 
就是了 
就要 
属于 
左右 
巨大 
巩固 
已 
已矣 
已经 
巴 
巴巴 
帮助 
并不 
并不是 
广大 
广泛 
应当 
应用 
应该 
庶乎 
庶几 
开展 
引起 
强烈 
强调 
归齐 
当前 
当地 
当时 
形成 
彻底 
彼时 
往往 
後来 
後面 
得了 
得出 
得到 
心里 
必然 
必要 
怎奈 
怎麽 
总是 
总结 
您们 
您是 
惟其 
意思 
愿意 
成为 
我是 
我的 
或则 
或曰 
战斗 
所在 
所幸 
所有 
所谓 
扩大 
掌握 
接著 
数/ 
整个 
方便 
方面 
无 
无法 
既往 
明显 
明确 
是不是 
是以 
是否 
显然 
显著 
普通 
普遍 
曾 
曾经 
替代 
最 
最后 
最大 
最好 
最後 
最近 
最高 
有利 
有力 
有及 
有所 
有效 
有时 
有点 
有的是 
有着 
有著 
末##末 
本地 
来自 
来说 
构成 
某某 
根本 
欢迎 
欤 
正值 
正在 
正巧 
正常 
正是 
此地 
此处 
此时 
此次 
每个 
每天 
每年 
比及 
比较 
没奈何 
注意 
深入 
清楚 
满足 
然後 
特别是 
特殊 
特点 
犹且 
犹自 
现代 
现在 
甚且 
甚或 
甚至于 
用来 
由是 
由此 
目前 
直到 
直接 
相似 
相信 
相反 
相同 
相对 
相应 
相当 
相等 
看出 
看到 
看看 
看见 
真是 
真正 
眨眼 
矣乎 
矣哉 
知道 
确定 
种 
积极 
移动 
突出 
突然 
立即 
竟而 
第二 
类如 
练习 
组成 
结合 
继后 
继续 
维持 
考虑 
联系 
能否 
能够 
自后 
自打 
至今 
至若 
致 
般的 
良好 
若夫 
若果 
范围 
莫不然 
获得 
行为 
行动 
表明 
表示 
要求 
规定 
觉得 
譬喻 
认为 
认真 
认识 
许多 
设或 
诚如 
说明 
说来 
说说 
诸 
诸如 
谁人 
谁料 
贼死 
赖以 
距 
转动 
转变 
转贴 
达到 
迅速 
过去 
过来 
运用 
还要 
这一来 
这次 
这点 
这种 
这般 
这麽 
进入 
进步 
进行 
适应 
适当 
适用 
逐步 
逐渐 
通常 
造成 
遇到 
遭到 
遵循 
避免 
那般 
那麽 
部分 
采取 
里面 
重大 
重新 
重要 
针对 
问题 
防止 
附近 
限制 
随后 
随时 
随著 
难道说 
集中 
需要 
非特 
非独 
若果 
中
这一
之间


'''