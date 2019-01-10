#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
import jieba
import re
from bs4 import BeautifulSoup
import os
import numpy as np
import codecs

a=np.array([1,2,3])
for i in a:
    print(i)

URl = "http://101.200.146.55/alipay/aliCashier/20170222000025835341"
# pattern = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
# result = re.findall(pattern,URl)
# print(len(result))
# print(result)
def traverse_directory(WebDirectory,mainfile):
    MD5_list = list()
    flag_list = list()
    URL_list = list()
    with open(mainfile,'r') as f:
        for i in f:
            flag=i.split(',',4)[1]
            if flag=='n':
                flag_list.append(1)
                MD5 = i.split(',', 4)[2]
                each_file = os.path.join(WebDirectory, MD5)
                MD5_list.append(each_file)
                URL_list.append(i.strip().split(',',4)[3])
            elif flag=='p':
                flag_list.append(0)
                MD5 = i.split(',', 4)[2]
                each_file = os.path.join(WebDirectory, MD5)
                MD5_list.append(each_file)
                URL_list.append(i.strip().split(',', 4)[3])
    return MD5_list,flag_list,URL_list
def read_file(filename):
    try:
        f=codecs.open(filename,'r',encoding='utf-8')
        Web_data=f.readlines()
        # print(filename)
        # print('******')
        Web_data = '\n'.join(Web_data)
        f.close()
    except:
        f=codecs.open(filename,'r',encoding='gb18030')
        Web_data = f.readlines()
        # print(Web_data)
        # print(filename)
        # print('******')
        Web_data = '\n'.join(Web_data)
        f.close()
    return Web_data
def get_dic():
    URLKeyword, URLchar, action, title = [], [], [], []
    with open('P_url_key_list','r') as f1:
        for i in f1:
            URLKeyword.append(i.strip())
    with open('URL_keyword.txt','r') as f2:
        for j in f2:
            URLchar.append(j.strip().split(',')[0].lower())
    with open('unique_action.txt','r') as f3:
        for z in f3:
            action.append(z.strip())
    with open('unique_title.txt','r') as f4:
        for h in f4:
            title.append(h.strip())
    return URLKeyword, URLchar, action, title
def URL_feature(data, URLKeyword, URLchar):
    # special_list = ['-', '.', '/', '@', '?', '&', '=', '_']
    print(URLKeyword)
    data = data.lower()
    # 钓鱼网站关键字特征（login，qrcode）
    URL_Pkey_list = [data.count(key) for key in URLKeyword]
    print(URL_Pkey_list)

    IPcheck = 0
    pattern = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
    result = re.findall(pattern, data)
    if len(result) > 0:
        IPcheck = 1

    http_result = 0
    if data.startswith('https://'):
        http_result = 1
        data = data[8:]
    elif data.startswith('http://'):
        data = data[7:]
    # 高频字母特征
    Char_Counts = [data.count(char) for char in URLchar]
    url_len = len(data)
    np_chars = np.asarray(Char_Counts)
    t1 = [np_chars.sum(), np_chars.max(), np_chars.std(), np_chars.mean(), sum(np_chars > 0)]
    num_chars_per = np_chars.sum() / float(url_len)
    t2 = np.asarray([IPcheck, http_result, url_len, num_chars_per] + Char_Counts)

    return np.hstack((t1, t2, np.asarray(URL_Pkey_list))), np.hstack((t1, t2))
def Web_feature(Web_data, title, action):
    P_Feature = [
        'alert',
        'register',
        'login',
        'qrcode',
        'javascript:alert_login('')'
    ]
    try:
        soup = BeautifulSoup(Web_data, "html.parser")
        # 关键词标签
        cf_count = [Web_data.count(cf) for cf in P_Feature]
        doc_length = len(Web_data)

        # ------- feature for p---------#
        inputs_h = soup.findAll('input', {'type': 'hidden'})
        inputs_b = soup.findAll('input', {'type': 'button'})
        buttons = soup.findAll('button')
        scripts = soup.findAll('script')
        imgs = soup.findAll('img')
        forms = soup.findAll('form', {'method': 'post'})

        scripts_len_list = [len(str(scr)) for scr in scripts]
        script_len = sum(scripts_len_list)
        script_len_per = script_len / float(doc_length)

        num_scripts = len(scripts)
        num_h_inputs = len(inputs_h)
        num_b_inputs = len(inputs_b)
        num_btns = len(buttons)
        num_img = len(imgs)
        num_forms = len(forms)

        p_t_list = [0] * (len(title.keys()) + 1)
        p_f_list = [0] * (len(action.keys()) + 1)

        if soup.title:
            t_title = soup.title.string
            if t_title:
                t_seg = jieba.cut(t_title)
                for seg in t_seg:
                    key_index = title.get(seg, -1)
                    p_t_list[key_index] += 1

        t1 = np.asarray(p_t_list)
        t_np = np.hstack((t1[:-1], [t1.sum(), t1.max(), t1.std(), t1.mean(), sum(t1 > 0)]))

        for i in forms:
            key_index = action.get(i.get('action'), -1)
            p_f_list[key_index] += 1

        t2 = np.asarray(p_f_list)
        f_np = np.hstack((t2[:-1], [t2.sum(), t2.max(), t2.std(), t2.mean(), sum(t2 > 0)]))

        other = np.asarray(
            [doc_length, script_len, script_len_per, num_scripts, num_h_inputs, num_b_inputs, num_btns, num_img,
             num_forms] + cf_count)
        return np.hstack((t_np, f_np, other))
    except:
        print("wrong")

if __name__ == "__main__":
    mainfile = './data/file_list_20170430_new的副本.txt'
    WebDirectory = './data/file的副本/'
    # URLKeyword, URLchar, action, title = get_dic()
    # # print(URLKeyword, URLchar)
    # feature1, feature2 = URL_feature(URl, URLKeyword, URLchar)
    # print(feature1.sum(),feature2.sum())
    # print(len(feature1.tolist()),len(feature2.tolist()))
    URLKeyword, URLchar, action, title = get_dic()
    MD5_list, flag_list, URL_list = traverse_directory(WebDirectory, mainfile)


