from transformers import T5Tokenizer, T5ForConditionalGeneration
from main import Template
from src.utils import *

args={'config':'config/config.yaml',
      'data_name':'laptops',
      'reasoning':'thor',
      'cuda_index':0,
      'zero_shot':False}
label_dic={'positive':0, 'negative':1, 'neutral':2}

path=r"D:\Coding\Pretrained\flan-t5-xl"
tokenizer = T5Tokenizer.from_pretrained(path)
model = T5ForConditionalGeneration.from_pretrained(path, device_map="auto", offload_folder='offload')

temp=Template(args)
train_data=temp.teacher()
train_data=train_data[:10]

step_1=[]
for each in train_data:
      tar=each[1]
      context_1,step_1_prompt=prompt_for_aspect_inferring(each[0],tar)
      step_1_output = model.generate(**step_1_prompt)

      context_2,step_2_prompt=prompt_for_opinion_inferring(context_1,tar,step_1_output)
      step_2_output = model.generate(**step_2_prompt)



      # arr=[t]
      # arr.extend(each[1:])
      # step_1.append(arr)


