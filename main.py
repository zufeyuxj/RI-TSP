import argparse
import yaml
import torch
from attrdict import AttrDict
import pandas as pd
import pickle as pkl
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import set_seed, load_params_LLM
from src.loader import MyDataLoader
from src.model import LLMBackbone,TeacherLLM
from src.engine import PromptTrainer, ThorTrainer

args={'config':'config/config.yaml',
      'data_name':'laptops',
      'reasoning':'thor',
      'cuda_index':0,
      'zero_shot':False}

class Template:
    def __init__(self, args):
        config = AttrDict(yaml.load(open(args['config'], 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        names = []
        for k, v in args.items():
            setattr(config, k, v)
        config.dataname = config.data_name
        set_seed(config.seed)
        # config.device = torch.device('cpu')
        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        names = [config.model_size, config.dataname] + names
        config.save_name = '_'.join(list(map(str, names))) + '_{}.pth.tar'
        self.config = config

    def teacher(self):
        # self.train_data=MyDataLoader(self.config).get_train_data()
        (self.trainLoader, self.validLoader, self.testLoader), self.config = MyDataLoader(self.config).get_data()

        # for data in self.trainLoader:
        #     print(data)
        teacher_path = r"D:\Coding\Pretrained\flan-t5-xl"
        teacher_model = TeacherLLM(self.config,teacher_path)
        self.config = load_params_LLM(self.config, teacher_model, self.trainLoader)
        trainer = ThorTrainer(teacher_model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        self.config.data_origin=False
        trainer.generate_data(self.trainLoader)

    def student(self):
        # for each in self.trainLoader:
        #     print(each)
        # self.trainLoader = MyDataLoader(self.config).get_student_data()
        self.config.data_origin = True
        self.trainLoader, self.validLoader, self.testLoader= MyDataLoader(self.config).get_student_data()

        # self.config.data_origin = False
        # self.config.reasoning = 'prompt'
        student_model=LLMBackbone(self.config).to(self.config.device)
        self.config = load_params_LLM(self.config, student_model, self.trainLoader)

        trainer=PromptTrainer(student_model,self.config,self.trainLoader,self.validLoader,self.testLoader)
        test_trainer = ThorTrainer(student_model, self.config, self.trainLoader, self.validLoader, self.testLoader)

        trainer.train()
        lines=trainer.lines

        # print(lines)

        # res=test_trainer.evaluate_step(self.testLoader)
        # print(res)
        # r=test_trainer.final_evaluate(epoch=12)
        # print(r)

        df = pd.DataFrame(lines)
        print(df.to_string())
        df.to_csv(args['data_name'] + '-student-' + self.config.model_size + '.csv')

        # for valid in self.validLoader:
        #     print(valid)
        #     batch_origin_context = [student_model.tokenizer.decode(ids) for ids in valid['input_ids']]
        #     break
        # for test in self.testLoader:
        #     batch_origin_context = [student_model.tokenizer.decode(ids) for ids in test['input_ids']]
        #     print(test)
        # return self.train_data
    def forward(self):

        (self.trainLoader, self.validLoader, self.testLoader), self.config = MyDataLoader(self.config).get_data()
        # 加载模型及其参数
        self.model = LLMBackbone(config=self.config).to(self.config.device)
        self.config = load_params_LLM(self.config, self.model, self.trainLoader)

        print(f"Running on the {self.config.data_name} data.")
        if self.config.reasoning == 'prompt':
            print("Choosing prompt one-step infer mode.")
            trainer = PromptTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'thor':
            print("Choosing thor multi-step infer mode.")
            trainer = ThorTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        else:
            raise 'Should choose a correct reasoning mode: prompt or thor.'

        if self.config.zero_shot == True:
            print("Zero-shot mode for evaluation.")
            r = trainer.evaluate_step(self.testLoader, 'test')
            print(r)
            return

        print("Fine-tuning mode for training.")
        trainer.train()
        lines = trainer.lines

        df = pd.DataFrame(lines)
        print(df.to_string())
        df.to_csv(args['data_name']+'-'+args['reasoning']+'-'+self.config.model_size+'.csv')

if __name__ == '__main__':

    # print(torch.cuda.is_available())
    # pass
    # 配置设置
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--cuda_index', default=0)
    # parser.add_argument('-r', '--reasoning', default='thor', choices=['prompt', 'thor'],
    #                     help='with one-step prompt or multi-step thor reasoning')
    # parser.add_argument('-z', '--zero_shot', action='store_true', default=True,
    #                     help='running under zero-shot mode or fine-tune mode')
    # parser.add_argument('-d', '--data_name', default='laptops', choices=['restaurants', 'laptops'],
    #                     help='semeval data name')
    # parser.add_argument('-f', '--config', default='./config/config.yaml', help='config file')
    # args = parser.parse_args()
    # for e in vars(args):
    #     print(e)


    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    template = Template(args)
    template.teacher()
    # template.student()
    # data = pkl.load(open('data/teacher2stu/laptops_google_flan-t5-xl.pkl', 'rb'))
    # print(data[0],data[1],data[2])


    pass

