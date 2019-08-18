import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable


class VLSTMModel(nn.Module):

    def __init__(self, args, infer=False):
        super(VLSTMModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # 保存数据
        self.rnn_size = args.rnn_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds = args.maxNumPeds
        self.seq_length = args.seq_length
        self.gru = args.gru

        # LSTM 节点
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

        # 输入层
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # 输出层
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # 其他单元
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, *args):
        # Construct the output variable
        input_data = args[0]
        hidden_states = args[1]
        cell_states = args[2]

        if self.gru:
            cell_states = None

        PedsList = args[3]
        num_pedlist = args[4]
        dataloader = args[5]
        look_up = args[6]

        numNodes = len(look_up)
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:
            outputs = outputs.cuda()

        # 序列中每一帧
        for framenum, frame in enumerate(input_data):

            nodeIDs_boundary = num_pedlist[framenum]
            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]
            # 如果没有则直接到下一帧
            if len(nodeIDs) == 0:
                continue

            # 节点列表
            list_of_nodes = [look_up[x] for x in nodeIDs]
            # 当前节点
            corr_index = Variable((torch.LongTensor(list_of_nodes)))

            # 选取当前节点输入位置信息
            nodes_current = frame[list_of_nodes, :]
            # 获取当前节点列表状态
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)
            cell_states_current = torch.index_select(cell_states, 0, corr_index)

            # 输入层
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))

            h_nodes, c_nodes = self.cell(input_embedded, (hidden_states_current, cell_states_current))
            # 输出层
            outputs[framenum * numNodes + corr_index.data] = self.output_layer(h_nodes)

            # 更新将节点状态
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes

        # 重构输出
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum * numNodes + node, :]

        return outputs_return, hidden_states, cell_states
