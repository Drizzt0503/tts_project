import os
import sys
import numpy as np
import torch
import argparse
import time
from opencc import OpenCC
from scipy.io import wavfile

import utils
from text import symbols,symbols_c, symbols_t
from text import cleaned_text_to_sequence
from text import cleaned_text_to_sequence_c,cleaned_text_to_sequence_t
from text import VITS_PinYin
from text import tailao_to_text
from vits_core.models import SynthesizerEval
from vits_core.models import SynthesizerEval_no_bert
from vits_core.models import SynthesizerTrn_no_bert_sdp,SynthesizerTrn_no_bert,SynthesizerTrn_multilingual


lib_dir = os.path.dirname(os.path.realpath(__file__))


class Text_normalizer():
    pass


class vits_c():
    def __init__(self,device,modelPath):
        self.device = torch.device(device)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # pinyin
        self.tts_front = VITS_PinYin(lib_dir+"/models/prosody_bert_model", self.device)
		# prosody_bert
        #self.prosody_bert = text_to_prosody()
        # config
        self.config = lib_dir+'/config/vits.json'
        self.hps = utils.get_hparams_from_file(self.config)
        # model
        self.net_g = SynthesizerEval_no_bert(
                    len(symbols_c),
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    **self.hps.model)
        # model_path = "logs/bert_vits/G_200000.pth"
        # utils.save_model(net_g, "vits_bert_model.pth")
        self.model=modelPath
        utils.load_model(self.model, self.net_g)
        self.net_g.eval()
        self.net_g.to(device)
        self.cc = OpenCC('tw2s')


    def _save_wav(self,wav, path, rate):
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        wavfile.write(path, rate, wav.astype(np.int16))

    def infer(self,path,text,pace):
        item=text
        item=self.cc.convert(item)
        try:
            phonemes, char_embeds = self.tts_front.chinese_to_phonemes(item)
        except:
            return 1, 'not in dic'
        input_ids = cleaned_text_to_sequence_c(phonemes)
        with torch.no_grad():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(self.device)
            x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths, noise_scale=0.5,
                                length_scale=pace)[0][0, 0].data.cpu().float().numpy()
        print('infer done')
        self._save_wav(audio, path, self.hps.data.sampling_rate)
        print('save done')
        return 0, 'ok'


class vits_e():
    def __init__(self,device,modelPath):
        self.device = torch.device(device)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # pinyin
        self.tts_front = VITS_PinYin(lib_dir+"/models/prosody_bert_model", self.device)
		# prosody_bert
        #self.prosody_bert = text_to_prosody()
        # config
        self.config = lib_dir+'/config/vits.json'
        self.hps = utils.get_hparams_from_file(self.config)
        # model
        self.net_g = SynthesizerEval_no_bert(
                    len(symbols),
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    **self.hps.model)
        # model_path = "logs/bert_vits/G_200000.pth"
        # utils.save_model(net_g, "vits_bert_model.pth")
        self.model=modelPath
        utils.load_model(self.model, self.net_g)
        self.net_g.eval()
        self.net_g.to(device)
        #self.cc = OpenCC('tw2s')


    def _save_wav(self,wav, path, rate):
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        wavfile.write(path, rate, wav.astype(np.int16))

    def infer(self,path,text,pace):
        text_norm = text_to_sequence(text, ["english_cleaners2"])
        text_norm = torch.LongTensor(text_norm)

        with torch.no_grad():
            x_tst = torch.LongTensor(text_norm).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([text_norm.size(0)]).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths, noise_scale=0.5,
                                length_scale=pace)[0][0, 0].data.cpu().float().numpy()
        print('infer done')
        self._save_wav(audio, path, self.hps.data.sampling_rate)
        print('save done')
        return 0, 'ok'

class vits_e_sdp():
    def __init__(self,device,modelPath):
        self.device = torch.device(device)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # pinyin
        self.tts_front = VITS_PinYin(lib_dir+"/models/prosody_bert_model", self.device)
		# prosody_bert
        #self.prosody_bert = text_to_prosody()
        # config
        self.config = lib_dir+'/config/sdp_vits.json'
        self.hps = utils.get_hparams_from_file(self.config)
        # model
        self.net_g = SynthesizerTrn_no_bert_sdp(
                    len(symbols),
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    **self.hps.model)
        # model_path = "logs/bert_vits/G_200000.pth"
        # utils.save_model(net_g, "vits_bert_model.pth")
        self.model=modelPath
        utils.load_model(self.model, self.net_g)
        self.net_g.eval()
        self.net_g.to(device)
        #self.cc = OpenCC('tw2s')


    def _save_wav(self,wav, path, rate):
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        wavfile.write(path, rate, wav.astype(np.int16))

    def infer(self,path,text,pace):
        text_norm = text_to_sequence(text, ["english_cleaners2"])
        text_norm = torch.LongTensor(text_norm)

        with torch.no_grad():
            x_tst = torch.LongTensor(text_norm).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([text_norm.size(0)]).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths, noise_scale=0.6,noise_scale_w=0.8,
                                length_scale=pace)[0][0, 0].data.cpu().float().numpy()
        print('infer done')
        self._save_wav(audio, path, self.hps.data.sampling_rate)
        print('save done')
        return 0, 'ok'
class vits_t():
    def __init__(self,device,modelPath):
        self.device = torch.device(device)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # pinyin
        self.tts_front = VITS_PinYin(lib_dir+"/models/prosody_bert_model", self.device)
		# prosody_bert
        #self.prosody_bert = text_to_prosody()
        # config
        self.config = lib_dir+'/config/vits.json'
        self.hps = utils.get_hparams_from_file(self.config)
        # model
        self.net_g = SynthesizerEval_no_bert(
                    len(symbols_t),
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    **self.hps.model)
        # model_path = "logs/bert_vits/G_200000.pth"
        # utils.save_model(net_g, "vits_bert_model.pth")
        self.model=modelPath
        utils.load_model(self.model, self.net_g)
        self.net_g.eval()
        self.net_g.to(device)
        #self.cc = OpenCC('tw2s')


    def _save_wav(self,wav, path, rate):
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        wavfile.write(path, rate, wav.astype(np.int16))

    def infer(self,path,text,pace):
        text_norm = tailao_to_text(text)
        print(text_norm)
        text_norm = cleaned_text_to_sequence_t(text_norm)
        text_norm = torch.LongTensor(text_norm)

        with torch.no_grad():
            x_tst = torch.LongTensor(text_norm).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([len(text_norm)]).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths, noise_scale=0.5,
                                length_scale=pace)[0][0, 0].data.cpu().float().numpy()
        print('infer done')
        self._save_wav(audio, path, self.hps.data.sampling_rate)
        print('save done')
        return 0, 'ok'

class vits_multilingual():
    def __init__(self,device,modelPath,sid_path):
        self.device = torch.device(device)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.prosody_bert = text_to_prosody()
        # config
        self.config = lib_dir+'/config/vits_multi.json'
        self.hps = utils.get_hparams_from_file(self.config)
        # model
        self.net_g = SynthesizerTrn_multilingual(
                    len(symbols),
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    n_speakers=self.hps.data.n_speakers,
                    n_languages=self.hps.data.n_languages,
                    **self.hps.model)
        # model_path = "logs/bert_vits/G_200000.pth"
        # utils.save_model(net_g, "vits_bert_model.pth")
        self.model=modelPath
        utils.load_model(self.model, self.net_g)
        self.net_g.eval()
        self.net_g.to(device)
        sid=  torch.load(sid_path)  #dim(1,192)
        self.sid=torch.squeeze(sid,0)    #dim(192)
        #self.cc = OpenCC('tw2s')


    def _save_wav(self,wav, path, rate):
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        wavfile.write(path, rate, wav.astype(np.int16))

    def infer(self,path,text,pace,blid,lid_list):
        text_norm = text
        text_norm = cleaned_text_to_sequence(text_norm)
        text_norm = torch.LongTensor(text_norm)
        lid_l=len(lid_list)
        lidd=torch.LongTensor(lid_list)
        lidd=lidd.unsqueeze(0).to(self.device)
        blidd=torch.LongTensor(blid)
        blidd=blidd.to(self.device)

        with torch.no_grad():
            ssid=self.sid.unsqueeze(0).to(self.device)
            x_tst = torch.LongTensor(text_norm).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([len(text_norm)]).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths,sid=ssid,lid=lidd,lid_lengths=lid_l
            ,b_lid=blidd, noise_scale=0.5,length_scale=pace)[0][0, 0].data.cpu().float().numpy()
        print('infer done')
        self._save_wav(audio, path, self.hps.data.sampling_rate)
        print('save done')
        return 0, 'ok'

def TW_sep(text):
    all_list= text.split('|tw>')
    return_list=[]
    if len(all_list) % 2 == 1:
        for i in range(len(all_list)):
            if i % 2 == 1:
                return_list.append([all_list[i],2])
            else:
                return_list.append([all_list[i],99])
        return return_list
    else:
        print('tw seprator error!')
def merge_same_lang(t_list):
    f_list=[]
    strm=''
    label=t_list[0][1]
    for ele in t_list:
        if ele[1] == label:
            strm += ele[0]
        else:
            f_list.append([strm,label])
            strm=ele[0]
            label=ele[1]
    f_list.append([strm,label])
    return f_list

def merge_diff_lang(t_list):
    if t_list[0][1] == 100:
        t_list[1][0] = t_list[0][0]+t_list[1][0]
        t_list.pop(0)
    f_list=[]
    for i,ele in enumerate(t_list):
        if ele[1] == 100:
            f_list.pop()
            f_list.append([t_list[i-1][0]+t_list[i][0],t_list[i-1][1]])
        else:
            f_list.append(ele)
    return f_list
def CH_EN_sep(text):
    index_list=[]
    for i,uchar in enumerate(text):
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
            index_list.append([uchar,1])
        elif uchar >= u'\u0041' and uchar <= u'\u005a':
            index_list.append([uchar,0])
        elif uchar >= u'\u0061' and uchar <= u'\u007a':
            index_list.append([uchar,0])
        else:
            index_list.append([uchar,100])
    f_list = merge_same_lang(index_list)
    f2_list = merge_diff_lang(f_list)
    f3_list = merge_same_lang(f2_list)
    return f3_list


def CH_phonemizer(text):
    from opencc import OpenCC
    from pypinyin import pinyin, lazy_pinyin, Style
    from text.phoneme_table import pinyin_dict
    cc=OpenCC('tw2s')
    text = cc.convert(text)
    text2 = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    t_str=''
    for any in text2:
        if any[:-1] in pinyin_dict:
            t_str += pinyin_dict[any[:-1]][0]+'_c '+pinyin_dict[any[:-1]][1]+any[-1]+'_c '
        else:
            t_str += 'sp_c '
    return t_str

def EN_phonemizer(text):
    import text as tx
    from text.phoneme_table import _punctuation as _punc
    temp = tx._clean_text(text, ["english_cleaners2"])
    abc = temp.split()
    t_str= ''
    for word in abc:
        for sym in word:
            t_str+=sym+' '
        t_str+= 'sp_e '
    return t_str
def handle_text(text):
    sen_tw = TW_sep(text)
    sen_all=[]
    for any in sen_tw:
        if any[1] == 99:
            sen2=CH_EN_sep(any[0])
            for sen in sen2:
                sen_all.append(sen)
        else:
            sen_all.append(any)
    f_str = ''
    lid_list=[]
    for any in sen_all:
        if any[1] == 1:
            text2 = CH_phonemizer(any[0])
            f_str += text2
            a = text2.split()
            for i in range(len(a)):
                lid_list.append(1)
            #print('1',text2)
        if any[1] == 0:
            text3 = EN_phonemizer(any[0])
            f_str += text3
            a = text3.split()
            for i in range(len(a)):
                lid_list.append(0)
            #print('2',text3)
        if any[1] == 2:
            f_str += any[0]
            a = any[0].split()
            for i in range(len(a)):
                lid_list.append(2)
    return f_str,lid_list


if __name__ == '__main__':
    #text="雷聲滾滾，閃電常明，山妖亦是戰戰兢兢，卻忍不住心中貪婪，時常躲在樹後，望向那座雷電風雨間的破廟。"
    #text="Hello, this is your mother fucker!"
    #text = "sin1-ni5 beh4 kau3--ah4, lan2 lai5-khi3 be2 ni5-tshai3 ho2--bo5?"
    #text = 'sim1-kuann1 an2-ne1 to3 siunn7-tsau2, ka1-ki7 na7 tsin1 u7-huat4-too7 tso3 siau3-kui7 to7 ho2--ah4, khi2-be2 siu1-hing7 m7-bian2 thau1-thau1-bong1.'
    #text1=['^_c', 'uei4_c', 'sh_c', 'en2_c', 'm_c', 'e5_c', 't_c', 'ai2_c', '^_c', 'uan1_c', 'r_c', 'en2_c', 'sh_c', 'ou3_c', 'j_c', 'i1_c', 'd_c', 'ou1_c', 'm_c', 'ai3_c', 'ˈ', 'a', 'ɪ', 'f', 'o', 'ʊ', 'n', '^_c', 'a5_c', 'sp_c', 'k_t', 'am2_t', 'conn_t', 's_t', 'i7_t', 'l_t', 'ong2_t', 'ts_t', 'iok4_t', '^_t', 'u7_t', 'conn_t', 'ts_t', 'inn5_t', 'conn_t', 'conn_t', '^_t', 'e5_t', 'split_t']
    #text='sil_c '
    #for any in text1:
    #    text+=any+' '
    #text+='sil_c'
    #print(text)
    #print(len(text1))
    #lid_list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]+[0,0,0,0,0,0,0]+[1,1,1]+[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]+[1]

    #print(len(lid_list))
    #===chinese no bert===
    #aaa=vits_c("cuda",'./temp/no_bert/G_100000.pth')    #model pace do not use yet
    #a,b = aaa.infer('./hello.wav',text,1.0)
    #===english===
    #aaa=vits_e("cuda",'./temp/vits_eng/G_160000.pth')    #model pace do not use yet
    #a,b = aaa.infer('./hello.wav',text,1.0)
    #===english sdp===
    #aaa=vits_e_sdp("cuda",'./temp/vits_eng_2/G_160000.pth')    #model pace do not use yet
    #a,b = aaa.infer('./hello.wav',text,1.0)
    #===taiwanese===
    #aaa=vits_t("cuda",'./temp/vits_taiwan/G_80000.pth')    #model pace do not use yet
    #a,b = aaa.infer('./hello.wav',text,0.8)
    #print(a,b)
    #===multilingual===
    #emb_path ='./temp/backup2_train_set/embed/SSB11250084.pt'
    #text = 'sil_e ð ə sp_e k ˈ o ː ɹ s sp_e ɹ ˈ ɛ k ɚ d sp_e w ʌ z sp_e ɪ n sp_e h ɪ z sp_e n ˈ e ɪ m . sp_e sil_e'
    #emb_path ='./temp/train_set/embed/SSB18630056.pt'
    #aaa=vits_multilingual("cuda",'./temp/multilingual/G_360000.pth',emb_path)    #model pace do not use yet
    #a,b = aaa.infer('./hello.wav',text,1.0)
    #print(a,b)a
    #===multilingual_final===
    #emb_path ='./temp/train_set/embed/000209.pt'
    #aaa=vits_multilingual("cuda",'./temp/multilingual_josh/G_6000.pth',emb_path)    #model pace do not use yet
    #a,b = aaa.infer('./hello.wav',text,1.0,[1],lid_list)
    #print(a,b)a

    main_lang = 1
    t_text_list = ['k_t', 'am2_t', 'conn_t', 's_t', 'i7_t', 'l_t', 'ong2_t', 'ts_t', 'iok4_t', '^_t', 'u7_t', 'conn_t', 'ts_t', 'inn5_t', 'conn_t', 'conn_t', '^_t', 'e5_t', 'split_t']
    t_text=''
    for any in t_text_list:
        t_text+=any+' '
    #text=f"零件費用產生,|tw>{t_text}|tw>服務人員apple pie將另行報價,請問您iphone接受嗎?It was all too little, too late."
    text=f"零件費用產生Trump,服務人員hello kitty將另行報價,請問您遠東A棟B棟C棟接受嗎?"
    #text = 'If you can make it there, you can make it anywhere.'
    #text = 'Apple'
    print(text)
    #text2=TW_sep(text)
    #print(text2)
    #text3=what_lang(text)
    #print(text3)
    textf,lid_list=handle_text(text)
    if main_lang == 1:
        textf='sil_c '+textf+'sil_c'
        lid_list.insert(0,1)
        lid_list.append(1)
    if main_lang == 0:
        textf='sil_e '+textf+'sil_e'
        lid_list.insert(0,0)
        lid_list.append(0)
    print(textf)
    print(len(lid_list))
    print(lid_list)
    emb_path ='./temp/train_set/embed/000209.pt'
    #emb_path ='./temp/backup2_train_set/embed/SSB11250084.pt'
    #aaa=vits_multilingual("cuda",'./temp/multilingual_emb/G_380000.pth',emb_path)    #model pace do not use yet
    aaa=vits_multilingual("cuda",'./temp/multilingual_josh/G_6000.pth',emb_path)    #model pace do not use yet
    a,b = aaa.infer('./hello.wav',textf,1.0,[main_lang],lid_list)
    print(a,b)

