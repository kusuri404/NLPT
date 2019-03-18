import logging
import os
import pickle
import re

class posseg(object):
    def __init__(self, model_path=''):
        self.state_list = ['B', 'M', 'E', 'S']
        self.pos_list = ['ag', 'a', 'ad', 'an', 'b', 'bg', 'c', 'dg', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                    'mg', 'ng', 'n', 'nr', 'ns', 'nt', 'nx', 'nz', 'o', 'p', 'q', 'r', 'rg', 's', 'tg', 't', 'u', 'vg', 'v',
                    'vd', 'vn', 'w', 'x', 'y', 'z', 'zother']
        self.state_pos_list = []
        for state in self.state_list:
            self.state_pos_list.extend([state+'_'+pos for pos in self.pos_list])
        self.load_para = False
        self.model_init()
        if model_path != '':
            self.load_model(model_path)

    def model_init(self):
        self.A_dict = {}
        self.B_dict = {}
        self.Pi_dict = {}
        self.Char_set = {}

    def load_model(self, model_path):
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.A_dict = pickle.load(f)
                self.B_dict = pickle.load(f)
                self.Pi_dict = pickle.load(f)
                self.Char_set = pickle.load(f)
                self.load_para = True
            self. load_para = True
            logging.info('model loaded!')
        else:
            logging.warning('model load failed!')

    def seve_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.A_dict, f)
            pickle.dump(self.B_dict, f)
            pickle.dump(self.Pi_dict, f)
            pickle.dump(self.Char_set, f)

    def train(self, corpus_path, update=False):
        self.model_init()
        count_dic = {}

        def init_parameters():
            for state in self.state_pos_list:
                self.A_dict[state] = {s:0.0 for s in self.state_pos_list}
                self.B_dict[state] = {}
                count_dic[state] = 0
            for state in self.state_list:
                self.Pi_dict[state] = 0.0

        def word_pos_split(text):
            parts = text.split('/')
            word = ''.join(parts[:-1])
            pos = parts[-1].lower()
            if pos not in self.pos_list:
                print(text, pos,'->other')
                pos = 'zother'
            # assert pos in self.pos_list, 'unknown part of speech'
            return word, pos

        def makeLabel(text, pos):
            out_text = []
            if len(text) == 1:
                out_text.append('S'+'_'+pos)
            else:
                out_text += ['B'+'_'+pos] + ['M'+'_'+pos] * (len(text) - 2) + ['E'+'_'+pos]

            return out_text

        if not update:
            init_parameters()
        line_num = -1

        words = set()
        with open(corpus_path, encoding='utf8') as f:
            for line in f:
                line_num += 1

                line = line.strip()
                if not line:
                    continue

                word_part = re.findall(r'\[(.+?)\]',line)
                for wp in word_part:
                    combine_word = [word_pos_split(i) for i in wp.split(' ')]
                    combine_word = ''.join([i[0] for i in combine_word])
                    line = line.replace('['+wp+']', combine_word+'/')

                word_pos_list = [word_pos_split(i) for i in line.split(' ')]
                word_list = [i[0] for i in word_pos_list]
                char_list = ''.join(word_list)
                words |= set([i for i in char_list])

                stateList = []
                for w in word_pos_list:
                    stateList.extend(makeLabel(w[0], w[-1]))

                assert len(char_list) == len(stateList)
                eps = 1e-6

                for k,v in enumerate(stateList):
                    count_dic[v] += 1
                    if k == 0:
                        self.Pi_dict[v[0]] += 1
                    else:
                        self.A_dict[stateList[k - 1]][v] += 1
                        self.B_dict[stateList[k]][char_list[k]] = self.B_dict[stateList[k]].get(char_list[k], 0) + 1.0

        self.Pi_dict = {k: v * 1.0 / line_num for k, v in self.Pi_dict.items()}
        self.A_dict = {k: {k1: v1 / (count_dic[k]+eps) for k1, v1 in v.items()} for k, v in self.A_dict.items()}
        self.B_dict = {k: {k1: (v1 + 1) / (count_dic[k]+eps) for k1, v1 in v.items()} for k, v in self.B_dict.items()}
        self.Char_set = words

        self.load_para = True
        return self

    def viterbi(self, text, states, start_p, trans_p, emit_p):
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y[0]] * emit_p[y].get(text[0], 0)
            path[y] = [y]

        for t in range(1, len(text)):
            V.append({})
            newpath = {}

            neverSeen = text[t] not in self.Char_set
            if neverSeen:
                print('[warning] %s is not in the dictionary.'%text[t])

            for y in states:
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0
                (prob, state) = max([(V[t-1][y0] * trans_p[y0].get(y, 0) * emitP, y0) for y0 in states])
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath

        # if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
        #     (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', 'M')])
        # else:
        (prob, state) = max([(V[len(text) - 1][y], y) for y in states])

        return prob, path[state]

    def cut(self, text):
        if not self.load_para:
            self.load_model('./default_HMM.model')
        prob, pos_list = self.viterbi(text, self.state_pos_list, self.Pi_dict, self.A_dict, self.B_dict)
        begin, end = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i][0]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin:i+1], pos_list[i][2:]
                end = i + 1
            elif pos == 'S':
                yield char, pos_list[i][2:]
                end = i + 1
        if end < len(text):
            yield text[end:], pos_list[-1][2:]
        #     yield char,pos_list[i]