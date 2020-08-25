
from __future__ import absolute_import, division, print_function

import textcnn
import argparse
import glob
import logging
import os
import random
import textcnn

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME,
                          BertConfig,
                          BertForSequenceClassification,
                          BertTokenizer,
                          XLMConfig,
                          XLMForSequenceClassification,
                          XLMTokenizer,
                          XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer
                          )

from transformers import AdamW, WarmupLinearSchedule

from dataset_utils import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)
import setGPU
from data_util import WordSubstitude
import json
import pickle
import string


logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def randomize_testset(args, perturb_pca, random_smooth, similarity_threshold, perturbation_constraint):
    
    data_dir = args.data_dir
    skip = args.skip
    num_random_sample = args.num_random_sample

    if args.task_name == 'imdb':
        dataset_name = 'imdb'
    elif args.task_name == 'amazon':
        dataset_name = 'amazonfull'
    
    folder1 = os.path.join(data_dir, "test", "pos")
    folder2 = os.path.join(data_dir, "test", "neg")

    # creating the skipped test set
    path_of_data = os.path.join(data_dir, 'test', "pos" + str(skip))
    if not os.path.exists(path_of_data):
        os.makedirs(path_of_data)

        path_list = os.listdir(folder1)
        path_list.sort()
        count = 0
        for filename in tqdm(path_list):
            if count % skip == 0:
                x_raw = open(os.path.join(folder1, filename)).read()
                a=open(os.path.join(path_of_data, filename), 'w')
                a.write(x_raw)
            count += 1
    
    path_of_data = os.path.join(data_dir, 'test', "neg" + str(skip))
    if not os.path.exists(path_of_data):
        os.makedirs(path_of_data)

        path_list = os.listdir(folder2)
        path_list.sort()
        count = 0
        for filename in tqdm(path_list):
            if count % skip == 0:
                x_raw = open(os.path.join(folder2, filename)).read()
                a=open(os.path.join(path_of_data, filename), 'w')
                a.write(x_raw)
            count += 1
    
    folder1 = os.path.join(data_dir, "test", "pos" + str(skip))
    folder2 = os.path.join(data_dir, "test", "neg" + str(skip))

    # creating folder for randomized testset
    out_data_dir = os.path.join(data_dir, 'random_test_' + str(similarity_threshold) + '_' + str(perturbation_constraint), "pos" + str(skip))
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    print('Generating randomized data for pos label')
    path_list = os.listdir(folder1)
    for filename in tqdm(path_list):
        #if count % skip == 0:
        data_name = os.path.splitext(filename)[0]
        data = open(os.path.join(folder1, filename)).read()
        if data:
            path_of_data = os.path.join(out_data_dir, data_name)
            if not os.path.exists(path_of_data):
                os.makedirs(path_of_data)
            for _ in range(num_random_sample):
                data_perturb = str(random_smooth.get_perturbed_batch(np.array([[data]]))[0][0])
                a=open(os.path.join(path_of_data, data_name + '_' + str(_) + '.txt'), 'w')
                a.write(data_perturb)
                    
            examples = randomized_create_examples_from_folder(path_of_data, 1)
            torch.save(examples, path_of_data + '/example')

    # creating folder for randomized testset
    print('Generating randomized data for neg label')
    out_data_dir = os.path.join(data_dir, 'random_test_'+str(similarity_threshold) + '_' + str(perturbation_constraint), "neg" + str(skip))
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    path_list = os.listdir(folder2)
    for filename in tqdm(path_list):
        data_name = os.path.splitext(filename)[0]
        data = open(os.path.join(folder2, filename)).read()
        if data:
            path_of_data = os.path.join(out_data_dir, data_name)
            if not os.path.exists(path_of_data):
                os.makedirs(path_of_data)
            for _ in range(num_random_sample):
                data_perturb = str(random_smooth.get_perturbed_batch(np.array([[data]]))[0][0])
                a=open(os.path.join(path_of_data, data_name + '_' + str(_) + '.txt'), 'w')
                a.write(data_perturb)
                    
            examples = randomized_create_examples_from_folder(path_of_data, 0)
            torch.save(examples, path_of_data + '/example')

    tv_table_dir = os.path.join(data_dir, dataset_name + '_counterfitted_tv_pca' + str(similarity_threshold) + '_' + str(perturbation_constraint) + '.pkl')

    if not os.path.exists(tv_table_dir):
        tv_table = calculate_tv_table(args, perturb_pca)
        

def randomized_create_examples_from_folder(folder, label):
        """Creates examples for the training and dev sets from labelled folder."""
        examples = []
        i = 0
        for input_file in os.listdir(folder):
            if input_file.endswith(".txt"):
                with open(os.path.join(folder, input_file), "r") as f:
                    tem_text = f.readlines()
                    if tem_text:
                        text_a=tem_text[0]
                        guid = "%s-%d" % ('test', i); i+=1
                        text_b = None
                        label = str(label)
                        examples.append(
                                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

def load_and_cache_examples(args, folder, task, tokenizer):
    processor = processors[task]()
    output_mode = output_modes[task]
    
    cached_features_file = os.path.join(folder, 'cached_{}_{}_{}_{}_{}'.format(
        'test',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task),
        str(args.similarity_threshold)))
        
    if os.path.exists(cached_features_file):
        #if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        examples = torch.load(folder + '/example')
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            #logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def randomized_evaluate(args, model, tokenizer, prefix=""):
    if args.task_name == 'imdb':
        dataset_name = 'imdb'
    elif args.task_name == 'amazon':
        dataset_name = 'amazonfull'
    # read tv table
    tv_table_dir = os.path.join(args.data_dir, dataset_name + '_counterfitted_tv_pca' + str(args.similarity_threshold) + '_' + str(args.perturbation_constraint) + '.pkl')
    pkl_file = open(tv_table_dir, 'rb')
    tv_table = pickle.load(pkl_file)
    pkl_file.close()
    
    total_cert_acc = 0
    cert_acc_count = 0
    
    tv = calculate_tv(tv_table)

    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.checkpoint_dir,)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        results[eval_task] = {}
        
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        
        random_data_dir_pos = os.path.join(args.data_dir, 'random_test_' + str(args.similarity_threshold) + '_' + str(args.perturbation_constraint), 'pos' + str(args.skip))
        random_data_dir_neg = os.path.join(args.data_dir, 'random_test_' + str(args.similarity_threshold) + '_' + str(args.perturbation_constraint), 'neg' + str(args.skip))   
        
        text_count = 0
        if os.path.exists(random_data_dir_pos):
            files = os.listdir(random_data_dir_pos)
            for file in tqdm(files):

                # read original text
                original_text_dir = os.path.join(args.data_dir, 'test', 'pos', file+'.txt')
                original_text = open(original_text_dir).read()

                eval_dataset = load_and_cache_examples(args, os.path.join(random_data_dir_pos, file), eval_task, tokenizer)
                
                args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
                eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
                eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
                
                eval_loss = 0.0
                nb_eval_steps = 0
                preds = None
                out_label_ids = None
                for batch in eval_dataloader:
                    model.eval()
                    batch = tuple(t.to(args.device) for t in batch)

                    with torch.no_grad():
                        inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                        'labels':         batch[3]}
                        outputs = model(**inputs)

                        tmp_eval_loss, logits = outputs[:2]

                        eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                    if preds is None:
                        preds = logits.detach().cpu().numpy()
                        out_label_ids = inputs['labels'].detach().cpu().numpy()
                    else:
                        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                        out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

                eval_loss = eval_loss / nb_eval_steps
                if args.output_mode == "classification":
                    preds = np.argmax(preds, axis=1)
                elif args.output_mode == "regression":
                    preds = np.squeeze(preds)
                result = compute_metrics(eval_task, preds, out_label_ids)

                results[eval_task][file] = {'text': original_text, 'similarity_threshold': args.similarity_threshold, 'p': result['acc'], 'label': 'pos'}
                
                tem_tv = tv.get_tv(original_text)

                if result['acc'] - 1. + np.prod(tem_tv[0:20]) >= 0.5 + args.mc_error:
                    total_cert_acc += 1.
                
                cert_acc_count += 1.
                
                if text_count % 10 == 0:
                    print('certified acc: ', total_cert_acc/cert_acc_count)
                    if not os.path.exists(args.result_dir):
                        os.makedirs(args.result_dir)

                    result_save_name = os.path.join(args.result_dir, 'cached_{}_{}_{}_{}_{}_{}'.format(
                        'test',
                        list(filter(None, args.model_name_or_path.split('/'))).pop(),
                        str(args.max_seq_length),
                        str(args.task_name),
                        str(args.similarity_threshold),
                        str(args.perturbation_constraint)))
    
                    output = open(result_save_name, 'wb')
                    pickle.dump(results, output)
                    output.close()

                text_count += 1

        if os.path.exists(random_data_dir_neg):
            files = os.listdir(random_data_dir_neg)
            for file in tqdm(files):

                # read original text
                original_text_dir = os.path.join(args.data_dir, 'test', 'neg', file+'.txt')
                original_text = open(original_text_dir).read()

                eval_dataset = load_and_cache_examples(args, os.path.join(random_data_dir_neg, file), eval_task, tokenizer)
                
                args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
                # Note that DistributedSampler samples randomly
                eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
                eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
                
                eval_loss = 0.0
                nb_eval_steps = 0
                preds = None
                out_label_ids = None
                for batch in eval_dataloader:
                    model.eval()
                    batch = tuple(t.to(args.device) for t in batch)

                    with torch.no_grad():
                        inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                        'labels':         batch[3]}
                        outputs = model(**inputs)

                        tmp_eval_loss, logits = outputs[:2]

                        eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                    if preds is None:
                        preds = logits.detach().cpu().numpy()
                        out_label_ids = inputs['labels'].detach().cpu().numpy()
                    else:
                        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                        out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

                eval_loss = eval_loss / nb_eval_steps
                if args.output_mode == "classification":
                    preds = np.argmax(preds, axis=1)
                elif args.output_mode == "regression":
                    preds = np.squeeze(preds)
                result = compute_metrics(eval_task, preds, out_label_ids)

                results[eval_task][file] = {'text': original_text, 'similarity_threshold': args.similarity_threshold, 'p': result['acc'], 'label': 'neg'}
                
                tem_tv = tv.get_tv(original_text)
                if result['acc'] - 1. + np.prod(tem_tv[0:20]) >= 0.5 + args.mc_error:
                    total_cert_acc += 1.
                
                cert_acc_count += 1.

                if text_count % 10 == 0:
                    print('certified acc: ', total_cert_acc/cert_acc_count)
                    if not os.path.exists(args.result_dir):
                        os.makedirs(args.result_dir)

                    result_save_name = os.path.join(args.result_dir, 'cached_{}_{}_{}_{}_{}_{}'.format(
                        'test',
                        list(filter(None, args.model_name_or_path.split('/'))).pop(),
                        str(args.max_seq_length),
                        str(args.task_name),
                        str(args.similarity_threshold),
                        str(args.perturbation_constraint)))
    
                    output = open(result_save_name, 'wb')
                    pickle.dump(results, output)
                    output.close()


                text_count += 1
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    result_save_name = os.path.join(args.result_dir, 'cached_{}_{}_{}_{}_{}_{}'.format(
        'test',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.task_name),
        str(args.similarity_threshold),
        str(args.perturbation_constraint)))
    
    output = open(result_save_name, 'wb')
    pickle.dump(results, output)
    output.close()
    print('certified acc: ', total_cert_acc/cert_acc_count)

class calculate_tv:
    def __init__(self, table_tv):
        self.table = table_tv
        self.key_dict = set(table_tv.keys())
        self.exclude = set(string.punctuation)
    
    def get_tv(self, text):
        # input a text (string)
        tem_text = text.split(' ')
        tv_list = np.zeros(len(tem_text))
        if tem_text[0]:
            for j in range(len(tem_text)):
                if tem_text[j][-1] in self.exclude:
                    tem_text[j] = tem_text[j][0:-1]
                if tem_text[j] in self.key_dict:
                    tv_list[j] = self.table[tem_text[j]]
                else:
                    tv_list[j] = 1.
        return np.sort(tv_list)

def calculate_tv_table(args, perturb):
    similarity_threshold = args.similarity_threshold
    data_dir = args.data_dir

    if args.task_name == 'imdb':
        dataset_name = 'imdb'
    elif args.task_name == 'amazon':
        dataset_name = 'amazonfull'

    # reading vocabulary
    pkl_file = open(os.path.join(data_dir, dataset_name + '_vocab_pca.pkl'), 'rb')
    data_vocab = pickle.load(pkl_file)
    pkl_file.close()

    # reading neighbor set
    pkl_file = open(os.path.join(data_dir, dataset_name + '_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'), 'rb')
    data_neighbor = pickle.load(pkl_file)
    pkl_file.close()

    data_neighbor = data_neighbor['neighbor']

    total_intersect = 0
    total_freq = 0

    counterfitted_tv = {}
    for key in tqdm(data_neighbor.keys()):
        if not key in perturb.keys():
            counterfitted_tv[key] = 1

            total_intersect += data_vocab[key]['freq']*1
            total_freq += data_vocab[key]['freq']

        elif perturb[key]['isdivide'] == 0:
            counterfitted_tv[key] = 1
        
            total_intersect += data_vocab[key]['freq']*1
            total_freq += data_vocab[key]['freq']

        else:
            key_neighbor = data_neighbor[key]
            cur_min = 10.
            num_perb = len(perturb[key]['set'])
            for neighbor in key_neighbor:
                num_neighbor_perb = len(perturb[neighbor]['set'])
                num_inter_perb = len(list(set(perturb[neighbor]['set']).intersection(set(perturb[key]['set']))))
                tem_min = num_inter_perb/num_perb
                if tem_min < cur_min:
                    cur_min = tem_min
            counterfitted_tv[key] = cur_min

            total_intersect += data_vocab[key]['freq']*cur_min
            total_freq += data_vocab[key]['freq']

    Name = os.path.join(data_dir, dataset_name + '_counterfitted_tv_pca' + str(similarity_threshold) + '_' + str(args.perturbation_constraint) + '.pkl')
    output = open(Name, 'wb')
    pickle.dump(counterfitted_tv, output)
    output.close()
    print('calculate total variation finishes')
    print('-'*89)

    return counterfitted_tv

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

task_labels = {
    "neg": "0",
    "pos": "1",
}


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--checkpoint_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--result_dir", default=None, type=str, required=True,
                        help="The output directory where the result will be written.")   
    
    parser.add_argument('--net_type', default='bert', type=str,
                        help='networktype: bert, textcnn, and so on')

    parser.add_argument("--skip", default=20, type=int,
                        help="Evaluate one testing point every skip testing point")
    parser.add_argument("--num_random_sample", default=2000, type=int,
                        help="The number of random samples of each text.")
    parser.add_argument("--similarity_threshold", default=0.7, type=float,
                        help="The similarity constraint to be considered as synonym.")
    parser.add_argument("--perturbation_constraint", default=800, type=int,
                        help="The maximum size of perturbation set of each word")
    parser.add_argument("--mc_error", default=0.01, type=float,
                        help="Monte Carlo Error based on concentration inequality")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    args = parser.parse_args()

    if os.path.exists(args.checkpoint_dir) and os.listdir(args.checkpoint_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.checkpoint_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.net_type == 'textcnn':
        model = textcnn.Model()
    else:
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Distributed and parallel training
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    similarity_threshold = args.similarity_threshold
    perturbation_constraint = args.perturbation_constraint
    similarity_threshold = similarity_threshold

    if args.task_name == 'imdb':
        dataset_name = 'imdb'
    elif args.task_name == 'amazon':
        dataset_name = 'amazonfull'

    pkl_file = open(args.data_dir + dataset_name + '_perturbation_constraint_pca' + str(similarity_threshold) + '_' + str(perturbation_constraint) + '.pkl', 'rb')
    perturb_pca = pickle.load(pkl_file)
    pkl_file.close()

    # shorten the perturbation set to desired length constraint
    for key in perturb_pca.keys():
        if len(perturb_pca[key]['set'])>perturbation_constraint:

            tem_neighbor_count = 0
            tem_neighbor_list = []
            for tem_neighbor in perturb_pca[key]['set']:
                tem_neighbor_list.append(tem_neighbor)
                tem_neighbor_count += 1
                if tem_neighbor_count >= perturbation_constraint:
                    break
            perturb_pca[key]['set'] = tem_neighbor_list
            perturb_pca[key]['isdivide'] = 1 
           
    random_smooth = WordSubstitude(perturb_pca)

    # generate randomized data

    randomize_testset(args, perturb_pca, random_smooth, similarity_threshold, perturbation_constraint)

    # Evaluation
    if args.local_rank in [-1, 0]:
        checkpoints = [args.checkpoint_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.checkpoint_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            if args.net_type == 'bert':
                model = model_class.from_pretrained(checkpoint)
            else:
                model = textcnn.Model()
                checkpoint = torch.load(os.path.join(checkpoint, 'checkpoint.pth.tar'))
                model.load_state_dict(checkpoint['state_dict'])

            model.to(args.device)
            
            randomized_evaluate(args, model, tokenizer, prefix=global_step)

if __name__ == "__main__":
    main()
