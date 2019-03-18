import os
import logging
import pickle

class MM(object):
    MM = 'MM'
    IMM = 'IMM'
    BIMM = 'BiMM'

    def __init__(self, dicPath, maxlen_ratio=1):
        self.dictionary = set()
        self.maxLen = 0
        # assert len_ratio<=1
        with open(dicPath, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                self.maxLen = max(len(line), self.maxLen)
        self.maxLen = int(self.maxLen * maxlen_ratio)

    def cutMM(self,text):
        result = []
        word_num = 0
        succ_index = 0
        index = 0
        text_len = len(text)
        while index < text_len:
            word = None
            for size in range(self.maxLen,0,-1):
                if index + size > text_len:
                    continue
                piece = text[index:(index+size)]
                if piece in self.dictionary:
                    if index > succ_index:
                        result.append(text[succ_index:index])
                        word_num += 1
                    word = piece
                    result.append(word)
                    succ_index = index + size
                    index += size
                    word_num += 1
                    break
            if word is None:
                index += 1
                if index == text_len:
                    result.append(text[succ_index:index])
                    word_num += 1

        return result, word_num

    def cutIMM(self, text):
        result = []
        word_num = 0
        succ_index = 0
        index = len(text)
        while index>0:
            word = None
            for size in range(self.maxLen,0,-1):
                if index - size < 0:
                    continue
                piece = text[(index-size):index]
                if piece in self.dictionary:
                    if index < succ_index:
                        result.append(text[index,succ_index])
                        word_num += 1
                    word = piece
                    result.append(word)
                    succ_index = index - size
                    index -= size
                    word_num += 1
                    break
            if word is None:
                index -= 1

        return result[::-1], word_num

    def cutBiMM(self, text):
        result, num1 = self.cutIMM(text)
        tmp, num2 = self.cutMM(text)
        if num2 < num1:
            result = tmp
            num1 = num2
        return result, num1

    def cut(self, text, direction = 'BiMM'):
        if direction == self.MM:
            result, _ = self.cutMM(text)
        elif direction == self.IMM:
            result, _ = self.cutIMM(text)
        elif direction == self.BIMM:
            result, _ = self.cutBiMM()

        return result

class HMM(object):
    def __init__(self, model_path=''):
        self.state_list = ['B', 'M', 'E', 'S']
        self.load_para = False
        self.model_init()
        if model_path != '':
            self.load_model(model_path)

    def model_init(self):
        self.A_dict = {}
        self.B_dict = {}
        self.Pi_dict = {}

    def load_model(self, model_path):
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.A_dict = pickle.load(f)
                self.B_dict = pickle.load(f)
                self.Pi_dict = pickle.load(f)
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

    def train(self, corpus_path, update=False):
        self.model_init()
        count_dic = {}

        def init_parameters():
            for state in self.state_list:
                self.A_dict[state] = {s:0.0 for s in self.state_list}
                self.Pi_dict[state] = 0.0
                self.B_dict[state] = {}
                count_dic[state] = 0

        def makeLabel(text):
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']

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

                word_list = [i for i in line if i != ' ']
                words |= set(word_list)

                phraseList = line.split()
                stateList = []
                for w in phraseList:
                    stateList.extend(makeLabel(w))

                assert len(word_list) == len(stateList)

                for k,v in enumerate(stateList):
                    count_dic[v] += 1
                    if k == 0:
                        self.Pi_dict[v] += 1
                    else:
                        self.A_dict[stateList[k - 1]][v] += 1
                        self.B_dict[stateList[k]][word_list[k]] = self.B_dict[stateList[k]].get(word_list[k], 0) + 1.0

        self.Pi_dict = {k: v * 1.0 / line_num for k, v in self.Pi_dict.items()}
        self.A_dict = {k: {k1: v1 / count_dic[k] for k1, v1 in v.items()} for k, v in self.A_dict.items()}
        self.B_dict = {k: {k1: (v1 + 1) / count_dic[k] for k1, v1 in v.items()} for k, v in self.B_dict.items()}

        self.load_para = True
        return self

    def viterbi(self, text, states, start_p, trans_p, emit_p):
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]

        for t in range(1, len(text)):
            V.append({})
            newpath = {}

            neverSeen = text[t] not in emit_p['S'].keys() and \
                text[t] not in emit_p['M'].keys() and \
                text[t] not in emit_p['E'].keys() and \
                text[t] not in emit_p['B'].keys()

            for y in states:
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0
                (prob, state) = max([(V[t-1][y0] * trans_p[y0].get(y, 0) * emitP, y0) for y0 in states if V[t-1][y0] > 0])
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
        prob, pos_list = self.viterbi(text, self.state_list, self.Pi_dict, self.A_dict, self.B_dict)
        begin, end = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin:i+1]
                end = i + 1
            elif pos == 'S':
                yield char
                end = i + 1
        if end < len(text):
            yield text[end:]

