import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

# 大模型（Flan-T5）的类
class LLMBackbone(nn.Module):
    def __init__(self, config):
        super(LLMBackbone, self).__init__()
        self.config = config
        self.engine = T5ForConditionalGeneration.from_pretrained(config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    # 前向传播计算损失
    def forward(self, **kwargs):
        input_ids, input_masks, output_ids, output_masks = [kwargs[w] for w in '\
        input_ids, input_masks, output_ids, output_masks'.strip().split(', ')]
        output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100
        output = self.engine(input_ids, attention_mask=input_masks, decoder_input_ids=None,
                             decoder_attention_mask=output_masks, labels=output_ids)
        loss = output[0]
        return loss

    # 输入prompt，通过FLan-T5模型生成下文语句
    def generate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks,
                                      max_length=self.config.max_length)
        dec = [self.tokenizer.decode(ids) for ids in output]
        output = [context.replace('<pad>', '').replace('</s>', '').strip() for context in dec]
        return output

    def evaluate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks, max_length=200)
        dec = [self.tokenizer.decode(ids) for ids in output]
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = [label_dict.get(w.replace('<pad>', '').replace('</s>', '').strip(), 0) for w in dec]
        return output

class TeacherLLM(nn.Module):
    def __init__(self, config,path):
        super(TeacherLLM, self).__init__()
        self.config=config
        self.engine = T5ForConditionalGeneration.from_pretrained(path, device_map="auto", offload_folder='offload')
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    # 输入prompt，通过FLan-T5模型生成下文语句
    def generate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks,
                                      max_length=self.config.max_length)
        dec = [self.tokenizer.decode(ids) for ids in output]
        output = [context.replace('<pad>', '').replace('</s>', '').strip() for context in dec]
        return output

    def evaluate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks, max_length=200)
        dec = [self.tokenizer.decode(ids) for ids in output]
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = [label_dict.get(w.replace('<pad>', '').replace('</s>', '').strip(), 0) for w in dec]
        return output
