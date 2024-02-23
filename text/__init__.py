from text.phoneme_table import symbols_c, symbols,symbols_t,tailao_dic,_pause_t


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id_c = {s: i for i, s in enumerate(symbols_c)}
_id_to_symbol_c = {i: s for i, s in enumerate(symbols_c)}


def cleaned_text_to_sequence_c(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = [_symbol_to_id_c[symbol] for symbol in cleaned_text.split()]
    return sequence


def sequence_to_text_c(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol_c[symbol_id]
        result += s
    return result
#=======================




import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer


class CharEmbedding(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.bert_config = BertConfig.from_pretrained(model_dir)
        self.hidden_size = self.bert_config.hidden_size
        self.bert = BertModel(self.bert_config)
        self.proj = nn.Linear(self.hidden_size, 256)
        self.linear = nn.Linear(256, 3)

    def text2Token(self, text):
        token = self.tokenizer.tokenize(text)
        txtid = self.tokenizer.convert_tokens_to_ids(token)
        return txtid

    def forward(self, inputs_ids, inputs_masks, tokens_type_ids):
        out_seq = self.bert(input_ids=inputs_ids,
                            attention_mask=inputs_masks,
                            token_type_ids=tokens_type_ids)[0]
        out_seq = self.proj(out_seq)
        return out_seq


class TTSProsody(object):
    def __init__(self, path, device):
        self.device = device
        self.char_model = CharEmbedding(path)
        self.char_model.load_state_dict(
            torch.load(
                os.path.join(path, 'prosody_model.pt'),
                map_location="cpu"
            ),
            strict=False
        )
        self.char_model.eval()
        self.char_model.to(self.device)

    def get_char_embeds(self, text):
        input_ids = self.char_model.text2Token(text)
        input_masks = [1] * len(input_ids)
        type_ids = [0] * len(input_ids)
        input_ids = torch.LongTensor([input_ids]).to(self.device)
        input_masks = torch.LongTensor([input_masks]).to(self.device)
        type_ids = torch.LongTensor([type_ids]).to(self.device)

        with torch.no_grad():
            char_embeds = self.char_model(
                input_ids, input_masks, type_ids).squeeze(0).cpu()
        return char_embeds

    def expand_for_phone(self, char_embeds, length):  # length of phones for char
        assert char_embeds.size(0) == len(length)
        expand_vecs = list()
        for vec, leng in zip(char_embeds, length):
            vec = vec.expand(leng, -1)
            expand_vecs.append(vec)
        expand_embeds = torch.cat(expand_vecs, 0)
        assert expand_embeds.size(0) == sum(length)
        return expand_embeds.numpy()
#========================
import re

import pypinyin
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin

import numpy as np

from text.phoneme_table import pinyin_dict


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def clean_chinese(text: str):
    text = text.strip()
    text_clean = []
    for char in text:
        if (is_chinese(char)):
            text_clean.append(char)
        else:
            if len(text_clean) > 1 and is_chinese(text_clean[-1]):
                text_clean.append(',')
    text_clean = ''.join(text_clean).strip(',')
    return text_clean


class VITS_PinYin:
    def __init__(self, bert_path, device):
        self.pinyin_parser = Pinyin(MyConverter())
        self.prosody = TTSProsody(bert_path, device)

    def chinese_to_phonemes(self, text):
        # @todo:考虑使用g2pw的chinese bert替换原始的pypinyin,目前测试下来运行速度太慢。
        # 将标准中文文本符号替换成 bert 符号库中的单符号,以保证bert的效果.
        text = text.replace("——", "...")\
            .replace("—", "...")\
            .replace("……", "...")\
            .replace("…", "...")\
            .replace('“', '"')\
            .replace('”', '"')\
            .replace("\n", "")
        tokens = self.prosody.char_model.tokenizer.tokenize(text)
        text = ''.join(tokens)
        assert not tokens.count("[UNK]")
        pinyins = np.reshape(pypinyin.pinyin(text, style=pypinyin.TONE3), (-1))
        try:
            phone_index = 0
            phone_items = []
            phone_items.append('sil')
            count_phone = []
            count_phone.append(1)
            temp = ""

            len_pys = len(tokens)
            for word in tokens:
                if is_chinese(word):
                    count_phone.append(2)
                    if (phone_index >= len_pys):
                        print(
                            f"!!!![{text}]plz check ur text whether includes MULTIBYTE symbol.\
                                (请检查你的文本中是否包含多字节符号)")
                    pinyin = pinyins[phone_index]
                    phone_index = phone_index + 1
                    if not pinyin[-1].isdigit():
                        pinyin += "5"
                    if pinyin[:-1] in pinyin_dict:
                        tone = pinyin[-1]
                        a = pinyin[:-1]
                        a1, a2 = pinyin_dict[a]
                        phone_items += [a1, a2 + tone]
                else:
                    temp += word
                    if temp == pinyins[phone_index]:
                        temp = ""
                        phone_index += 1
                    count_phone.append(1)
                    phone_items.append('sp')

            count_phone.append(1)
            phone_items.append('sil')
            phone_items_str = ' '.join(phone_items)
        except IndexError as e:
            print('except:', e)

        text = f'[PAD]{text}[PAD]'
        char_embeds = self.prosody.get_char_embeds(text)
        char_embeds = self.prosody.expand_for_phone(char_embeds, count_phone)
        return phone_items_str, char_embeds

""" from https://github.com/keithito/tacotron """
#from text import cleaners


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  clean_text = _clean_text(text, cleaner_names)
  for symbol in clean_text:
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id_t = {s: i for i, s in enumerate(symbols_t)}
_id_to_symbol_t = {i: s for i, s in enumerate(symbols_t)}


def cleaned_text_to_sequence_t(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = [_symbol_to_id_t[symbol] for symbol in cleaned_text.split(' ')]
    return sequence


def sequence_to_text_t(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol_t[symbol_id]
        result += s
    return result
#=======================

from 臺灣言語工具.解析整理.拆文分析器 import 拆文分析器
from 臺灣言語工具.音標系統.閩南語.臺灣閩南語羅馬字拼音 import 臺灣閩南語羅馬字拼音

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
    return p2list

