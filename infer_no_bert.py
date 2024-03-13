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
from vits_core.models import SynthesizerTrn_no_bert_sdp,SynthesizerTrn_no_bert


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

class vits_multi():
    def __init__(self,device,modelPath,sid_path):
        self.device = torch.device(device)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        sid=  torch.load(sid_path)  #dim(1,192)
        self.sid=torch.squeeze(sid,0)    #dim(192)
        #self.cc = OpenCC('tw2s')


    def _save_wav(self,wav, path, rate):
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        wavfile.write(path, rate, wav.astype(np.int16))

    def infer(self,path,text,pace):
        text_norm = text
        text_norm = cleaned_text_to_sequence(text_norm)
        text_norm = torch.LongTensor(text_norm)

        with torch.no_grad():
            ssid=self.sid.unsqueeze(0).to(self.device)
            x_tst = torch.LongTensor(text_norm).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([len(text_norm)]).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths,sid=ssid, noise_scale=0.5,
                                length_scale=pace)[0][0, 0].data.cpu().float().numpy()
        print('infer done')
        self._save_wav(audio, path, self.hps.data.sampling_rate)
        print('save done')
        return 0, 'ok'
if __name__ == '__main__':
    #text="零件費用產生，服務人員將另行報價，請問您接受嗎?"
    #text="雷聲滾滾，閃電常明，山妖亦是戰戰兢兢，卻忍不住心中貪婪，時常躲在樹後，望向那座雷電風雨間的破廟。"
    #text="Hello, this is your mother fucker!"
    #text = "sin1-ni5 beh4 kau3--ah4, lan2 lai5-khi3 be2 ni5-tshai3 ho2--bo5?"
    #text = 'sim1-kuann1 an2-ne1 to3 siunn7-tsau2, ka1-ki7 na7 tsin1 u7-huat4-too7 tso3 siau3-kui7 to7 ho2--ah4, khi2-be2 siu1-hing7 m7-bian2 thau1-thau1-bong1.'
    textl=['^_c', 'uei4_c', 'sh_c', 'en2_c', 'm_c', 'e5_c', 't_c', 'ai2_c', '^_c', 'uan1_c', 'r_c', 'en2_c', 'sh_c', 'ou3_c', 'j_c', 'i1_c', 'd_c', 'ou1_c', 'm_c', 'ai3_c', 'ˈ', 'a', 'ɪ', 'f', 'o', 'ʊ', 'n', '^_c', 'a5_c', 'sp_c', 'k_t', 'am2_t', 'conn_t', 's_t', 'i7_t', 'l_t', 'ong2_t', 'ts_t', 'iok4_t', '^_t', 'u7_t', 'conn_t', 'ts_t', 'inn5_t', 'conn_t', 'conn_t', '^_t', 'e5_t', 'split_t']
    text='sil_c '
    for any in textl:
        text+=any+' '
    text+='sil_c'
    print(text)
    
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
    #emb_path ='./temp/train_set/embed/SSB11250084.pt'
    emb_path ='./temp/train_set/embed/SSB18630056.pt'
    aaa=vits_multi("cuda",'./temp/multilingual/G_360000.pth',emb_path)    #model pace do not use yet
    a,b = aaa.infer('./hello.wav',text,1.0)
    print(a,b)


