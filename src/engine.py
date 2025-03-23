import os
import torch
import numpy as np
import torch.nn as nn
import pickle as pkl
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
from src.utils import prompt_for_opinion_inferring, prompt_for_polarity_inferring, prompt_for_polarity_label

# ***************最主要的部分在这一份代码里**************
# Self-Consistency的这部分代码并没有看到？

# Prompt模型训练类
class PromptTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.final_score = 0
        self.final_res = ''

        self.scores, self.lines = [], []
        self.re_init()

    def train(self):
        best_score, best_iter = 0, -1
        for epoch in tqdm(range(self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            self.train_step()
            result = self.evaluate_step(mode='valid')
            self.re_init()
            score = result['default']
            self.add_instance(result)

            res = self.get_best()

            if score > best_score:
                best_score, best_iter = score, epoch
                save_name = self.save_name.format(epoch)

                if not os.path.exists(self.config.target_dir):
                    os.makedirs(self.config.target_dir)
                torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict(), 'best_score': best_score},
                           save_name)
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
            self.model.to(self.config.device)

        res = self.final_evaluate(best_iter)
        score = res['default']
        self.add_instance(res)

        save_name = self.save_name.format(epoch)

        self.final_score, self.final_res = score, res

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader)
        losses = []
        for i, data in enumerate(train_data):
            loss = self.model(**data)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
            train_data.set_description(description)
            self.config.optimizer.step()
            self.config.scheduler.step()
            self.model.zero_grad()

    # 用于计算每一个batch的准确率并记录
    def evaluate_step(self, dataLoader=None, mode='valid'):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            with torch.no_grad():
                output = self.model.evaluate(**data)
                self.add_output(data, output)
        result = self.report_score(mode=mode)
        return result

    def final_evaluate(self, epoch=0):
        PATH = self.save_name.format(epoch)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        self.model.eval()
        res = self.evaluate_step(self.test_loader, mode='test')
        self.add_instance(res)
        return res

    def add_instance(self, res):
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax([w['default'] for w in self.lines])
        res = self.lines[best_id]
        return res

    def re_init(self):
        self.preds, self.golds = defaultdict(list), defaultdict(list)
        self.keys = ['total', 'explicits', 'implicits']

    def add_output(self, data, output):
        is_implicit = data['implicits'].tolist()
        gold = data['input_labels']
        for i, key in enumerate(self.keys):
            if i == 0:
                self.preds[key] += output
                self.golds[key] += gold.tolist()
            else:
                if i == 1:
                    ids = np.argwhere(np.array(is_implicit) == 0).flatten()
                else:
                    ids = np.argwhere(np.array(is_implicit) == 1).flatten()
                self.preds[key] += [output[w] for w in ids]
                self.golds[key] += [gold.tolist()[w] for w in ids]

    # 计算ACC F1等指标
    def report_score(self, mode='valid'):
        res = {}
        res['Acc_SA'] = accuracy_score(self.golds['total'], self.preds['total'])
        res['F1_SA'] = f1_score(self.golds['total'], self.preds['total'], labels=[0, 1, 2], average='macro')
        res['F1_ESA'] = f1_score(self.golds['explicits'], self.preds['explicits'], labels=[0, 1, 2], average='macro')
        res['F1_ISA'] = f1_score(self.golds['implicits'], self.preds['implicits'], labels=[0, 1, 2], average='macro')
        res['default'] = res['F1_SA']
        res['mode'] = mode
        for k, v in res.items():
            if isinstance(v, float):
                res[k] = round(v * 100, 3)
        return res

# Thor模型训练类
class ThorTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.final_score = 0
        self.final_res = ''
        self.scores, self.lines = [], []
        self.re_init()

    # 总训练过程
    def train(self):
        best_score, best_iter = 0, -1
        for epoch in tqdm(range(self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            # 每一步训练+验证集验证
            self.train_step()
            result = self.evaluate_step(mode='valid')
            self.re_init()
            score = result['default']
            self.add_instance(result)
            res = self.get_best()

            # 记录最好的一次epoch
            if score > best_score:
                best_score, best_iter = score, epoch
                save_name = self.save_name.format(epoch)
                if not os.path.exists(self.config.target_dir):
                    os.makedirs(self.config.target_dir)
                torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict(), 'best_score': best_score},
                           save_name)
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
            self.model.to(self.config.device)

        res = self.final_evaluate(best_iter)
        score = res['default']
        self.add_instance(res)
        save_name = self.save_name.format(epoch)
        self.final_score, self.final_res = score, res

    # 对应论文中的第二个prompt，即提问opinion的prompt
    def prepare_step_two(self, aspect_exprs, data):
        context_A_ids, target_ids = [data[w] for w in 'context_A_ids, target_ids'.strip().split(', ')]
        contexts_A = [self.model.tokenizer.decode(ids) for ids in context_A_ids]
        contexts_A = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_A]
        targets = [self.model.tokenizer.decode(ids) for ids in target_ids]
        targets = [context.replace('<pad>', '').replace('</s>', '').strip() for context in targets]

        new_prompts = []
        contexts_B = []
        for context, target, aspect_expr in zip(contexts_A, targets, aspect_exprs):
            context_B, prompt = prompt_for_opinion_inferring(context, target, aspect_expr)
            new_prompts.append(prompt)
            contexts_B.append(context_B)

        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=self.config.max_length)
        batch_inputs = batch_inputs.data
        batch_contexts_B = self.model.tokenizer.batch_encode_plus(contexts_B, padding=True, return_tensors='pt',
                                                                  max_length=self.config.max_length)
        batch_contexts_B = batch_contexts_B.data

        # 数据都以字典的形式进行保存
        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'context_B_ids': batch_contexts_B['input_ids'],
            'target_ids': target_ids,
        }

        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    # 对应论文中的第三个prompt，即提问polarity情感极性
    def prepare_step_three(self, opinion_exprs, data):
        context_B_ids, target_ids = [data[w] for w in 'context_B_ids, target_ids'.strip().split(', ')]
        contexts_B = [self.model.tokenizer.decode(ids) for ids in context_B_ids]
        contexts_B = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_B]
        targets = [self.model.tokenizer.decode(ids) for ids in target_ids]
        targets = [context.replace('<pad>', '').replace('</s>', '').strip() for context in targets]

        new_prompts = []
        contexts_C = []
        for context, target, opinion_expr in zip(contexts_B, targets, opinion_exprs):
            context_C, prompt = prompt_for_polarity_inferring(context, target, opinion_expr)
            new_prompts.append(prompt)
            contexts_C.append(context_C)

        batch_contexts_C = self.model.tokenizer.batch_encode_plus(contexts_C, padding=True, return_tensors='pt',
                                                                  max_length=self.config.max_length)
        batch_contexts_C = batch_contexts_C.data
        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=self.config.max_length)
        batch_inputs = batch_inputs.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'context_C_ids': batch_contexts_C['input_ids'],
        }
        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    # 这里是将情感极性再次进行prompt提示询问，以免第三步prompt出现一些并不属于情感极性三类的词
    def prepare_step_label(self, polarity_exprs, pre_cxt, data):
        output_ids, output_masks = [data[w] for w in 'output_ids, output_masks'.strip().split(', ')]
        context_C_ids = pre_cxt['context_C_ids']
        contexts_C = [self.model.tokenizer.decode(ids) for ids in context_C_ids]
        contexts_C = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_C]

        new_prompts = []
        for context_C, polarity_expr in zip(contexts_C, polarity_exprs):
            prompt = prompt_for_polarity_label(context_C, polarity_expr)
            new_prompts.append(prompt)

        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=3)
        batch_inputs = batch_inputs.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'output_ids': output_ids,
            'output_masks': output_masks,
        }
        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    # 训练所有数据的一步过程
    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader, total=self.train_loader.data_length)

        # 记录每一个batch数据下的loss值
        losses = []
        for i, data in enumerate(train_data):
            step_one_inferred_output = self.model.generate(**data)

            step_one_inferred_data = self.prepare_step_two(step_one_inferred_output, data)
            step_two_inferred_output = self.model.generate(**step_one_inferred_data)

            step_two_inferred_data = self.prepare_step_three(step_two_inferred_output, step_one_inferred_data)
            step_three_inferred_output = self.model.generate(**step_two_inferred_data)

            step_label_data = self.prepare_step_label(step_three_inferred_output, step_two_inferred_data, data)

            # 计算loss值仅使用了最后一步产生的，并没有像论文所说每一步都进行计算
            loss = self.model(**step_label_data)
            losses.append(loss.item())
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
            train_data.set_description(description)

            torch.cuda.empty_cache()
            self.config.optimizer.step()
            self.config.scheduler.step()
            self.config.optimizer.zero_grad()
            self.model.zero_grad()

    def generate_data(self,dataLoader=None, mode='valid'):
        self.model.eval()
        dataLoader = self.train_loader if dataLoader is None else dataLoader
        save_path='data/teacher2stu/'+self.config.data_name+'_'+self.config.teacher_model_path.replace('/','_')+".pkl"
        new_data=[]
        dataiter = dataLoader
        def generate(dataiter):
            new_train_data = []
            for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
                if i%20==0 and i!=0:
                    pkl.dump(new_data, open(save_path+'_'+str(i), 'wb'))
                # 先进行小批量测试
                # if i==10:
                #     break
                with torch.no_grad():
                    step_one_inferred_output = self.model.generate(**data)

                    step_one_inferred_data = self.prepare_step_two(step_one_inferred_output, data)
                    step_two_inferred_output = self.model.generate(**step_one_inferred_data)

                    step_two_inferred_data = self.prepare_step_three(step_two_inferred_output, step_one_inferred_data)
                    step_three_inferred_output = self.model.generate(**step_two_inferred_data)

                    step_label_data = self.prepare_step_label(step_three_inferred_output, step_two_inferred_data, data)
                    output = self.model.evaluate(**step_label_data)

                    # print(step_label_data['input_ids'])
                    batch_origin_context=[self.model.tokenizer.decode(ids) for ids in data['input_ids']]
                    origin_context = [context.replace('<pad>', '').replace('</s>', '').strip() for context in
                                         batch_origin_context]
                    batch_context = [self.model.tokenizer.decode(ids) for ids in step_two_inferred_data['input_ids']]
                    new_batch_context = [context.replace('<pad>', '').replace('</s>', '').strip() for context in batch_context]
                    batch_target=[self.model.tokenizer.decode(ids) for ids in data['target_ids']]
                    new_batch_target = [context.replace('<pad>', '').replace('</s>', '').strip() for context in
                                         batch_target]
                    new_batch_label = [int(ids) for ids in data['input_labels']]
                    new_batch_implicit = [int(ids) for ids in data['implicits']]
                    for i,answer in enumerate(output):
                        if answer==new_batch_label[i]:
                            new_d=[new_batch_context[i]+' [mask]',new_batch_target[i],new_batch_label[i],new_batch_implicit[i]]
                            new_train_data.append(new_d)
                        else:
                            idx=origin_context[i].find("which specific aspect of")
                            origin=origin_context[i][:idx]+'what is the sentiment polarity towards '+new_batch_target[i]+'? [mask]'
                            new_d=[origin,new_batch_target[i],new_batch_label[i],new_batch_implicit[i]]
                            new_train_data.append(new_d)
            return new_train_data

        self.config.is_raw=True
        test_data=[]
        for batch in self.test_loader:
            test_data.extend([each for each in batch])
        self.config.is_raw = False

        valid_data=[]
        for batch in self.valid_loader:
            valid_data.extend([each for each in batch])
        self.config.is_raw = False

        train_data=generate(dataiter)
        # dataLoader = self.valid_loader
        # dataiter = dataLoader
        # valid_data=generate(dataiter)

        new_data=[train_data,valid_data,test_data]
        # print(new_data)
        pkl.dump(new_data, open(save_path, 'wb'))


    # 计算每一个batch的输出和相关指标，做测试、验证时用
    def evaluate_step(self, dataLoader=None, mode='valid'):
        self.model.eval()
        self.config.data_origin=False
        self.config.reasoning='thor'
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            with torch.no_grad():
                step_one_inferred_output = self.model.generate(**data)

                step_one_inferred_data = self.prepare_step_two(step_one_inferred_output, data)
                step_two_inferred_output = self.model.generate(**step_one_inferred_data)

                step_two_inferred_data = self.prepare_step_three(step_two_inferred_output, step_one_inferred_data)
                step_three_inferred_output = self.model.generate(**step_two_inferred_data)

                step_label_data = self.prepare_step_label(step_three_inferred_output, step_two_inferred_data, data)
                output = self.model.evaluate(**step_label_data)
                self.add_output(data, output)

        result = self.report_score(mode=mode)
        return result

    # fine-tune中用于进行最后的测试集计算
    def final_evaluate(self, epoch=0):
        PATH = self.save_name.format(epoch)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        self.model.eval()
        res = self.evaluate_step(self.test_loader, mode='test')
        self.add_instance(res)
        return res

    def add_instance(self, res):
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax([w['default'] for w in self.lines])
        res = self.lines[best_id]
        return res

    def re_init(self):
        self.preds, self.golds = defaultdict(list), defaultdict(list)
        self.keys = ['total', 'explicits', 'implicits']

    def add_output(self, data, output):
        is_implicit = data['implicits'].tolist()
        gold = data['input_labels']
        for i, key in enumerate(self.keys):
            if i == 0:
                self.preds[key] += output
                self.golds[key] += gold.tolist()
            else:
                if i == 1:
                    ids = np.argwhere(np.array(is_implicit) == 0).flatten()
                else:
                    ids = np.argwhere(np.array(is_implicit) == 1).flatten()
                self.preds[key] += [output[w] for w in ids]
                self.golds[key] += [gold.tolist()[w] for w in ids]

    # 计算ACC F1等指标
    def report_score(self, mode='valid'):
        res = {}
        res['Acc_SA'] = accuracy_score(self.golds['total'], self.preds['total'])
        res['F1_SA'] = f1_score(self.golds['total'], self.preds['total'], labels=[0, 1, 2], average='macro')
        res['F1_ESA'] = f1_score(self.golds['explicits'], self.preds['explicits'], labels=[0, 1, 2], average='macro')
        res['F1_ISA'] = f1_score(self.golds['implicits'], self.preds['implicits'], labels=[0, 1, 2], average='macro')
        res['default'] = res['F1_SA']
        res['mode'] = mode
        for k, v in res.items():
            if isinstance(v, float):
                res[k] = round(v * 100, 3)
        return res
