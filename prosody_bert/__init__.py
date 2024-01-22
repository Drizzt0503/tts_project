from prosody_bert.ProsodyModel import TTSProsody


def text_to_prosody():

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
