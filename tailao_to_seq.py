import os
import csv
from 臺灣言語工具.解析整理.拆文分析器 import 拆文分析器
from 臺灣言語工具.音標系統.閩南語.臺灣閩南語羅馬字拼音 import 臺灣閩南語羅馬字拼音


_pun= "\"《ˋ》？)(’'〉”ˊ〈‘“"
_letters = 'abcdefghijklmnopqrstuvwxyz123456789 -,.?!;:─'
_letters2 = 'abcdefghijklmnopqrstuvwxyz123456789 '
_pass = _pun+_letters
"""
    abc = abc.replace('《',' ')\
        .replace('》',' ')\
        .replace('〈',' ')\
        .replace('〉',' ')\
        .replace('(',' ')\
        .replace(')',' ')\
        .replace('，',',')\
        .replace('。','.')\
        .replace('"',' ')\
        .replace('“',' ')\
        .replace('？','?')\
        .replace('”',' ')\
        .replace('‘',' ')\
        .replace('’',' ')
    abc = abc.replace('─',' split ')\
        .replace('……',' split split')\
        .replace('…',' split ')\
        .replace(',',' split ')\
        .replace('?',' split ')\
        .replace('!',' split ')\
        .replace(';',' split ')\
        .replace(':',' split ')\
        .replace('.',' split ')\
        .replace('-',' conn ')
    abc = abc.replace('   ',' ')\
            .replace('  ',' ')
"""

_pause_t = ["sil", "split", 'conn']
_init_t= [
    'p', 'ph', 'm', 'b',
    't', 'th', 'n', 'l',
    'k', 'kh', 'ng', 'g',
    'ts', 'tsh', 's', 'j',
    'h', '^'
    ]

_end_t = [
    'a', 'ah', 'ap', 'at', 'ak', 'ann', 'annh',
    'am', 'an', 'ang',
    'e', 'eh', 'enn', 'ennh',
    'i', 'ih', 'ip', 'it', 'ik', 'inn', 'innh',
    'im', 'in', 'ing',
    'o', 'oh',
    'oo', 'ooh', 'op', 'ok', 'om', 'ong', 'onn', 'onnh',
    'u', 'uh', 'ut', 'un',
    'ai', 'aih', 'ainn', 'ainnh',
    'au', 'auh', 'aunn', 'aunnh',
    'ia', 'iah', 'iap', 'iat', 'iak', 'iam', 'ian', 'iang', 'iann', 'iannh',
    'io', 'ioh', 'ie', 'ir',
    'iok', 'iong', 'ionn',
    'iu', 'iuh', 'iut', 'iunn', 'iunnh',
    'ua', 'uah', 'uat', 'uak', 'uan', 'uann', 'uannh',
    'ue', 'ueh', 'uenn', 'uennh',
    'ui', 'uih', 'uinn', 'uinnh',
    'iau', 'iauh', 'iaunn', 'iaunnh',
    'uai', 'uaih', 'uainn', 'uainnh',
    'm', 'mh', 'ng', 'ngh',
    'ioo', 'iooh'
    ]
    #add ie ir
_tones_t = ["1", "2", "3","4" ,"5", "7", "8", "9"]



tailao_dic={}
for i in _init_t:
    for j in _end_t:
        if i == '^':
            tailao_dic[j]=(i,j)
        else:
            tailao_dic[i+j]=(i,j)

#print(tailao_dic)

#text = "ngóo ti̍k sim luē kám-kak, jîn-sing ti̍k sím tiōng, put kánn lâi tín-tāng. ngóo put sī hònn tsú, mah put sī pháinn-lâng, ngóo tsí-sī ài bîn-bāng. ngóo put guān suî lōng suî hong, phiau lōng si tong, tshin-tshiūnn tsûn bû káng. ngóo put guān tsò-lâng, kan-khiáu tsuàn phāng, kam-guān lâi tsok ham lâng."
text = "sin1-ni5 beh4 kau3--ah4, lan2 lai5-khi3 be2 ni5-tshai3 ho2--bo5?"

def tailao_to_text(text):
    titem = 拆文分析器.建立句物件(text)
    ttext=titem.轉音(臺灣閩南語羅馬字拼音).看語句()
    abc = ttext.replace('─',' split ')\
        .replace('……',' split split')\
        .replace('…',' split ')\
        .replace(',',' split ')\
        .replace('?',' split ')\
        .replace('!',' split ')\
        .replace(';',' split ')\
        .replace(':',' split ')\
        .replace('.',' split ')\
        .replace('-',' conn ')
    abc = abc.replace('   ',' ')\
            .replace('  ',' ')
    plist=abc.strip().split(' ')
    p2list='sil '
    for che in plist:
        if che in _pause_t:
            p2list+=che+' '
        else:
            if che[:-1] in tailao_dic:
                a1,b1=tailao_dic[che[:-1]]
                p2list+=a1+' '
                p2list+=b1+che[-1]+' '
    p2list+='sil'
    print(p2list)

"""
for any in text3:
    plist=any[2].strip().split(' ')
    for che in plist:
        if che in _pause:
            pass
        else:
            if che[:-1] in tailao_dic:
                pass
            else:
                to_save=1
                print(any[0])
                print(che)
                print(plist)
                print('==============')
    if to_save == 0:
        p2=any[2].strip().split()
        p3=[]
        for pho in p2:
            if pho in _pause:
                p3.append(pho)
            elif pho[:-1] in tailao_dic:
                a1,b1 = tailao_dic[pho[:-1]]
                p3.append(a1)
                p3.append(b1+pho[-1])
            else:
                print('not in dic')
        p_str='sil '
        for ppho in p3:
            p_str+= ppho+ ' '
        p_str += 'sil'
        text4.append([any[0],any[1],p_str])
    to_save=0
"""

symbols_t=_pause_t+_init_t+[i + j for i in _end_t for j in _tones_t]

#print(symbols_t)
#print(len(symbols_t))

print(text)
print('=====')
tailao_to_text(text)
