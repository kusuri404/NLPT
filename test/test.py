from NLPT.wordcut import *
from NLPT.postag import *

# print('testing for MM...')
# cut_mm = MM('./dictionary.txt')
# print(cut_mm.cutMM('南京市长江大桥'))
# print(cut_mm.cutIMM('南京市长江大桥'))
# print(cut_mm.cutBiMM('南京市长江大桥'))
# print('end testing MM...')

# print('testing for HMM...')
# cut_hmm = HMM()
# # cut_hmm.train('./trainCorpus.txt_utf8')
# # cut_hmm.seve_model('./simple_HMM.model')
# # cut_hmm.load_model('./simple_HMM.model')
# res = cut_hmm.cut('我是一个好人，希望你能够理解')
# print(list(res))
# print('end testing HMM...')

# state_list = ['B', 'M', 'E', 'S']
# pos_list = ['ag', 'a', 'ad', 'an', 'b', 'c', 'dg', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
#             'ng', 'n', 'nr', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 's', 'tg', 't', 'u', 'vg', 'v',
#             'vd', 'vn', 'w', 'w', 'y', 'z']\
#
# state_pos_list =[]
# for state in state_list:
#     state_pos_list.extend([state+'_'+pos for pos in pos_list])
#
# print(state_pos_list)
# print(len(state_pos_list), len(state_list), len(pos_list))

print('testing.for postag...')
postagger = posseg()
# postagger.train('./people-daily.txt')
# postagger.seve_model('./default_pos.model')
postagger.load_model('./default_pos.model')
res = postagger.cut('这是一个非常棒的方案！')

print(list(res))
