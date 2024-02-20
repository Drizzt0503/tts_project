#import text
from text.symbols import symbols
from text import _symbol_to_id
import vits_pinyin

aaa=vits_pinyin.VITS_PinYin('./models/prosody_bert_model', 'cuda')

text= '哥哥目前测试下来运行速度太data慢paul'
text2 = '准备好接下这一招吗'
aaa.chinese_to_phonemes(text2)

#print(_symbol_to_id)
#print(symbols)
