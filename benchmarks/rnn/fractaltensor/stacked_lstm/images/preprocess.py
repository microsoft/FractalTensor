import sys

test_num = int(sys.argv[1])
print(f'test number : {test_num}')

data1 = open(f'stacked_lstm_unfused_elem_fused_bmm_{test_num}.csv',
             'r').read().rstrip().split('\n')
data2 = open(f'stacked_lstm_fused_elem_fused_bmm_{test_num}.csv',
             'r').read().rstrip().split('\n')

length = len(data1)
header = data1[0]

with open(f'stacked_lstm{test_num}.csv', 'w') as f:
    f.write('%s\n' % (header))
    for i in range(1, length, 2):
        unfused_elem_fused_bmm = data1[i]
        unfused_elem_fused_bmm = unfused_elem_fused_bmm.replace(
            'FractalTensor', 'FT_unfused-elem_fused-bmm')
        cudnn1 = float(data1[i + 1].split('|')[-2])

        # depth = int(data1[i].split('|')[2].replace('[', '').replace(
        #     ']', '').split(',')[-1])
        # if i >= 3 and depth % 2: continue

        fused_elem_fused_bmm = data2[i]
        fused_elem_fused_bmm = fused_elem_fused_bmm.replace(
            'FractalTensor', 'FT_fused-elem_fused-bmm')
        cudnn2 = float(data2[i + 1].split('|')[-2])

        cudnn = (cudnn1 + cudnn2) / 2.
        cudnn_str = data1[i + 1].split('|')
        cudnn_str = '|'.join(cudnn_str[0:-2]) + '|%.3f' % (cudnn) + '|'

        f.write('%s\n' % (unfused_elem_fused_bmm))
        f.write('%s\n' % (fused_elem_fused_bmm))
        f.write('%s\n' % (cudnn_str))
