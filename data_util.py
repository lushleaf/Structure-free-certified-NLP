import json
import numpy as np
import torch
import string

with open('counterfitted_neighbors.json', 'r') as f:
    ws = json.load(f)

class WordSubstitude:
    def __init__(self, table):
        self.table = table
        self.table_key = set(list(table.keys()))
        self.exclude = set(string.punctuation)

    def get_perturbed_batch(self, batch, rep=1):
        num_text = len(batch)
        out_batch = []
        for k in range(rep):
            for i in range(num_text):
                tem_text = batch[i][0].split(' ')
                if tem_text[0]:
                    for j in range(len(tem_text)):
                        if tem_text[j][-1] in self.exclude:
                            tem_text[j] = self.sample_from_table(tem_text[j][0:-1]) + tem_text[j][-1]
                        else:
                            tem_text[j] = self.sample_from_table(tem_text[j])
                #out_batch[k*num_text + i] = [' '.join(tem_text)]
                    out_batch.append([' '.join(tem_text)])
                else:
                    out_batch.append([batch[i][0]])
        return np.array(out_batch)

    def sample_from_table(self, word):
        if word in self.table_key:
            tem_words = self.table[word]['set']
            num_words = len(tem_words)
            index = np.random.randint(0, num_words)
            return tem_words[index]
        else:
            return word