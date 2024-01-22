import os
import sys
import numpy as np
import torch
import argparse
import time
from opencc import OpenCC
from scipy.io import wavfile

import utils
from text import symbols
from text import cleaned_text_to_sequence
from text import VITS_PinYin
from vits_core.models import SynthesizerEval
from vits_core.models import SynthesizerEval_temp


lib_dir = os.path.dirname(os.path.realpath(__file__))


class Text_normalizer():
    pass


class TTS_Vits():
    def __init__(self,device,modelPath):
        self.device = torch.device(device)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # pinyin
        self.tts_front = VITS_PinYin(lib_dir+"/models/prosody_bert_model", self.device)
		# prosody_bert
        self.prosody_bert = text_to_prosody()
        # config
        self.config = lib_dir+'/config/bert_vits.json'
        self.hps = utils.get_hparams_from_file(self.config)
        # model
        self.net_g = SynthesizerEval(
                    len(symbols),
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    **self.hps.model)
        # model_path = "logs/bert_vits/G_200000.pth"
        # utils.save_model(net_g, "vits_bert_model.pth")
        if modelPath == None:
            self.model = lib_dir + "/models/vits.pth"
        else:
            self.model='/home/yuhang/tts_model/'+modelPath
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
        phonemes, char_embeds = self.tts_front.chinese_to_phonemes(item)
        input_ids = cleaned_text_to_sequence(phonemes)
        with torch.no_grad():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(self.device)
            x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5,
                                length_scale=pace)[0][0, 0].data.cpu().float().numpy()
        print('infer done')
        self._save_wav(audio, path, self.hps.data.sampling_rate)
        print('save done')
# device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TTS_VitsM():
    def __init__(self,device,modelPath):
        self.device = torch.device(device)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # pinyin
        self.tts_front = VITS_PinYin(lib_dir+"/models/prosody_bert_model", self.device)
        # config
        self.config = lib_dir+'/config/vitsm.json'
        self.hps = utils.get_hparams_from_file(self.config)
        # model
        self.net_g = SynthesizerEval_temp(
                    len(symbols),
                    self.hps.data.filter_length // 2 + 1,
                    self.hps.train.segment_size // self.hps.data.hop_length,
                    **self.hps.model)
        # model_path = "logs/bert_vits/G_200000.pth"
        # utils.save_model(net_g, "vits_bert_model.pth")
        if modelPath == None :
            self.model = lib_dir + "/models/vitsm.pth"
            sid=torch.load(lib_dir+'/temp/josh_emb.pt')
        else :
            self.model='/home/yuhang/tts_model/'+modelPath
            s_emb = modelPath.rstrip('.pth')+'_emb.pt'            
            sid=  torch.load('/home/yuhang/tts_model/'+s_emb)  #dim(1,192)
        print(modelPath)            
        self.sid=torch.squeeze(sid,0)    #dim(192)
        utils.load_model(self.model, self.net_g)
        self.net_g.eval()
        self.net_g.to(device)
        self.cc = OpenCC('tw2s')

        #os.makedirs("./VitsM/vits_infer_out/", exist_ok=True)

    def _save_wav(self,wav, path, rate):
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        wavfile.write(path, rate, wav.astype(np.int16))

    def infer(self,path,text,pace):
        item=text
        item=self.cc.convert(item)
        phonemes, char_embeds = self.tts_front.chinese_to_phonemes(item)
        input_ids = cleaned_text_to_sequence(phonemes)
        with torch.no_grad():
            ssid=self.sid.unsqueeze(0).to(self.device)
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(self.device)
            x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(self.device)
            audio = self.net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, sid=ssid, noise_scale=0.5,length_scale=pace)[0][0, 0].data.cpu().float().numpy()
        print('infer done')
        self._save_wav(audio, path, self.hps.data.sampling_rate)
        print('save done')
if __name__ == '__main__':
    text="零件費用產生，服務人員將另行報價，請問您接受嗎?"
    print(text)
    #tts_serv=TTS_Vits("cuda",None)
    #aaa.set_parameters()#fs, not implement yet
    #infer(save_file_path, input_text, speech_speed)
    #tts_serv.infer('./hello.wav',text,0.8)
    aaa=TTS_VitsM("cuda",None)    #model pace do not use yet
    aaa.infer('./hello.wav',text,1.0)

