import os
import math
import torch
import numpy as np
import pickle as pkl
from src.utils import prompt_direct_inferring, prompt_direct_inferring_masked, prompt_for_aspect_inferring
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import random


# 数据集的类
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data_length = 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# 每个数据的初始形式：[句子，实体，情感极性，是否为implicit情感]
class MyDataLoader:
    def __init__(self, config):
        self.config = config
        config.preprocessor = Preprocessor(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # 获取最原始的训练数据
    def get_student_data(self):
        cfg = self.config
        path = 'data/teacher2stu/' + self.config.data_name+'_'+self.config.teacher_model_path.replace('/','_')+".pkl"
        # path = 'data/teacher2stu/' + 'train.pkl'
        if os.path.exists(path):
            self.data = pkl.load(open(path, 'rb'))
        else:
            assert False
        train_data,valid_data,test_data = self.data[:]
        load_data = lambda dataset: DataLoader(MyDataset(dataset), num_workers=0, worker_init_fn=self.worker_init,
                                               shuffle=self.config.shuffle, batch_size=self.config.batch_size,
                                               collate_fn=self.collate_fn)
        train_loader,valid_loader,test_loader= map(load_data, [train_data,valid_data,test_data])
        train_loader.data_length, valid_loader.data_length, test_loader.data_length = math.ceil(
            len(train_data) / self.config.batch_size), \
                                                                                      math.ceil(
                                                                                          len(valid_data) / self.config.batch_size), \
                                                                                      math.ceil(
                                                                                          len(test_data) / self.config.batch_size)
        res = [train_loader, valid_loader, test_loader]
        return res

    # 获取训练，验证，测试集的数据
    def get_data(self):
        cfg = self.config
        path = os.path.join(self.config.preprocessed_dir,
                            '{}_{}_{}.pkl'.format(cfg.data_name, cfg.model_size, cfg.model_path).replace('/', '-'))
        if os.path.exists(path):
            self.data = pkl.load(open(path, 'rb'))
        else:
            self.data = self.config.preprocessor.forward()
            pkl.dump(self.data, open(path, 'wb'))

        # 原始论文代码↓↓↓↓↓↓
        train_data, valid_data, test_data = self.data[:3]
        self.config.word_dict = self.data[-1]
        load_data = lambda dataset: DataLoader(MyDataset(dataset), num_workers=0, worker_init_fn=self.worker_init, \
                                               shuffle=self.config.shuffle, batch_size=self.config.batch_size,
                                               collate_fn=self.collate_fn)
        # 构建torch里的loader以传入后续数据
        train_loader, valid_loader, test_loader = map(load_data, [train_data, valid_data, test_data])
        train_loader.data_length, valid_loader.data_length, test_loader.data_length = math.ceil(
            len(train_data) / self.config.batch_size), \
                                                                                      math.ceil(
                                                                                          len(valid_data) / self.config.batch_size), \
                                                                                      math.ceil(
                                                                                          len(test_data) / self.config.batch_size)

        res = [train_loader, valid_loader, test_loader]
        # 返回三个数据loader和一个config
        return res, self.config

    # 用于整理数据输入
    def collate_fn(self, data):
        # 每个数据的形式：[句子，实体，情感极性标签，是否为implicit情感]
        input_tokens, input_targets, input_labels, implicits = [e[0] for e in data], [e[1] for e in data], \
                                                               [e[2] for e in data], [e[3] for e in data]
        # input_tokens, input_targets, input_labels, implicits = zip(*data)
        if self.config.is_raw:
            return data
        # 获取最原始数据的input_id等
        if self.config.data_origin:
            new_tokens = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                new_tokens.append(line)

            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits)
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            # 返回batch中已经将token转化为id的所有数据（包括attention_mask）
            return res

        # 使用prompt的模式
        if self.config.reasoning == 'prompt':
            new_tokens = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                if self.config.zero_shot == True:
                    _, prompt = prompt_direct_inferring(line, input_targets[i])
                else:
                    _, prompt = prompt_direct_inferring_masked(line, input_targets[i])
                new_tokens.append(prompt)

            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits)
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            # 返回batch中已经将token转化为id的所有数据（包括attention_mask）
            return res

        # 使用论文中三层思维链的形式，进行数据处理以输入到模型中
        elif self.config.reasoning == 'thor':

            new_tokens = []
            contexts_A = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                context_step1, prompt = prompt_for_aspect_inferring(line, input_targets[i])
                contexts_A.append(context_step1)
                new_tokens.append(prompt)

            batch_contexts_A = self.tokenizer.batch_encode_plus(contexts_A, padding=True, return_tensors='pt',
                                                                max_length=self.config.max_length)
            batch_contexts_A = batch_contexts_A.data
            batch_targets = self.tokenizer.batch_encode_plus(list(input_targets), padding=True, return_tensors='pt',
                                                             max_length=self.config.max_length)
            batch_targets = batch_targets.data
            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'context_A_ids': batch_contexts_A['input_ids'],
                'target_ids': batch_targets['input_ids'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits)
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            # 返回batch中已经将token转化为id的所有数据（包括attention_mask）
            return res

        else:
            raise 'choose correct reasoning mode: prompt or thor.'


# 预处理数据使用的类
class Preprocessor:
    def __init__(self, config):
        self.config = config

    def read_file(self):
        dataname = self.config.dataname
        train_file = os.path.join(self.config.data_dir, dataname,
                                  '{}_Train_v2_Implicit_Labeled_preprocess_finetune.pkl'.format(dataname.capitalize()))
        test_file = os.path.join(self.config.data_dir, dataname,
                                 '{}_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'.format(dataname.capitalize()))
        train_data = pkl.load(open(train_file, 'rb'))
        test_data = pkl.load(open(test_file, 'rb'))
        ids = np.arange(len(train_data))
        np.random.shuffle(ids)
        lens = 150  # 验证集的数据量为150个
        valid_data = {w: v[-lens:] for w, v in train_data.items()}
        train_data = {w: v[:-lens] for w, v in train_data.items()}

        return train_data, valid_data, test_data

    def transformer2indices(self, cur_data):
        res = []
        for i in range(len(cur_data['raw_texts'])):
            text = cur_data['raw_texts'][i]
            target = cur_data['raw_aspect_terms'][i]
            implicit = 0
            if 'implicits' in cur_data:
                implicit = cur_data['implicits'][i]
            label = cur_data['labels'][i]
            implicit = int(implicit)
            res.append([text, target, label, implicit])
        return res

    def forward(self):
        modes = 'train valid test'.split()
        dataset = self.read_file()
        res = []
        for i, mode in enumerate(modes):
            data = self.transformer2indices(dataset[i])
            res.append(data)
        return res
