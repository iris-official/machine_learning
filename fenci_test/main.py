#encoding=utf-8
from __future__ import unicode_literals
import sys
sys.path.append("../")

# import jieba
import jieba.posseg
import jieba.analyse

print('='*40)
print('1. 分词')
print('-'*40)

s = "佛山市兆航国际货运代理"

seg_list = jieba.cut(s, cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut(s, cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 默认模式

seg_list = jieba.cut(s)
print(", ".join(seg_list))

seg_list = jieba.cut_for_search(s)  # 搜索引擎模式
print(", ".join(seg_list))


print('='*40)
print('3. 关键词提取')
print('-'*40)
print(' TF-IDF')
print('-'*40)


for x, w in jieba.analyse.extract_tags(s, withWeight=True):
    print('%s %s' % (x, w))

print('-'*40)
print(' TextRank')
print('-'*40)

for x, w in jieba.analyse.textrank(s, withWeight=True):
    print('%s %s' % (x, w))

# print('='*40)
# print('4. 词性标注')
# print('-'*40)
#
# words = jieba.posseg.cut("我爱北京天安门")
# for word, flag in words:
#     print('%s %s' % (word, flag))
#
# print('='*40)
# print('6. Tokenize: 返回词语在原文的起止位置')
# print('-'*40)
# print(' 默认模式')
# print('-'*40)
#
# result = jieba.tokenize('永和服装饰品有限公司')
# for tk in result:
#     print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
#
# print('-'*40)
# print(' 搜索模式')
# print('-'*40)
#
# result = jieba.tokenize('永和服装饰品有限公司', mode='search')
# for tk in result:
#     print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
