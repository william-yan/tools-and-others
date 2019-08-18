import torch
import numpy as np
from torch.autograd import Variable

import argparse
import os
import time
import pickle
import subprocess

from vlstm_model import VLSTMModel
from utils import DataLoader
from helper import *

def main():
    #设置参数
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=5,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='prediction length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=27,
                        help='Maximum Number of Pedestrians')

    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')

    parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')

    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')

    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')

    parser.add_argument('--num_validation', type=int, default=3,
                        help='Total number of validation dataset for validate accuracy')

    parser.add_argument('--freq_validation', type=int, default=1,
                        help='Frequency number(epoch) of validation using validation data')

    parser.add_argument('--freq_optimizer', type=int, default=8,
                        help='Frequency number(epoch) of learning decay for optimizer')

    args = parser.parse_args()

    # 启动训练函数
    train(args)

def train(args):

    # 导入数据
    validation_dataset_executed = False
    prefix = ''
    f_prefix = '.'
    dataloader = DataLoader(f_prefix, args.batch_size, args.seq_length, args.num_validation, forcePreProcess=True)

    # 设置名称
    method_name = "VANILLALSTM"
    model_name = "LSTM"
    save_tar_name = method_name + "_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name + "_gru_model_"


    # model directory
    save_directory = os.path.join(prefix, 'model/')

    # Save the arguments int the config file
    with open(os.path.join(save_directory, method_name, model_name, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, method_name, model_name, save_tar_name + str(x) + '.tar')
    # 定义模型
    net = VLSTMModel(args)

    # 优化器
    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
    learning_rate = args.learning_rate
    # 不知道干嘛的一些参数
    best_val_loss = 100
    best_val_data_loss = 100
    smallest_err_val = 100000
    smallest_err_val_data = 100000
    best_epoch_val = 0
    best_epoch_val_data = 0
    best_err_epoch_val = 0
    best_err_epoch_val_data = 0
    all_epoch_results = []

    args.freq_validation = np.clip(args.freq_validation, 0, args.num_epochs)
    validation_epoch_list = list(range(args.freq_validation, args.num_epochs + 1, args.freq_validation))
    validation_epoch_list[-1] -= 1

    # 开始训练
    for epoch in range(args.num_epochs):
        print('****************Training epoch beginning******************')
        if dataloader.additional_validation and (epoch-1) in validation_epoch_list:
            dataloader.switch_to_dataset_type(True)
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # 每一批数据：
        for batch in range(dataloader.num_batches):
            start = time.time()
            # 读取当前批数据
            x, y, d, numPedsList, PedsList, target_ids = dataloader.next_batch()
            loss_batch = 0

            # 对每一个序列：
            for sequence in range(dataloader.batch_size):
                x_seq, _, d_seq, numPedsList_seq, PedsList_seq, target_id= \
                    x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence], target_ids[sequence]
                # 获取文件名
                folder_name = dataloader.get_directory_name_with_pointer(d_seq)
                dataset_data = dataloader.get_dataset_dimension(folder_name)
                # 数据格式转化 [id, x, y]=>[x, y], lookup_seq:预测序列初始化值
                x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
                # 目标id的第一个位置数据
                target_id_values = x_seq[0][lookup_seq[target_id], 0:2]
                # 目标数量
                numNodes = len(lookup_seq)
                # 初始化隐藏层数据
                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))

                net.zero_grad()
                optimizer.zero_grad()
                # 将数据喂给网络
                outputs, _, _ = net(x_seq, hidden_states, cell_states, PedsList_seq, numPedsList_seq, dataloader,lookup_seq)
                # 计算损失数据，神经网络输出为二维高斯分布，这里是自定义函数，将二维高斯分布的期望值与实际值做差
                loss = Gaussian2DLikelihood(outputs, x_seq, PedsList_seq, lookup_seq)
                loss_batch += loss.item()

                # 反向传播
                loss.backward()

                # 更新网络
                optimizer.step()
            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

            print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(
                epoch * dataloader.num_batches + batch,
                args.num_epochs * dataloader.num_batches,
                epoch,
                loss_batch, end - start))
        loss_epoch /= dataloader.num_batches

        # 每一个批次过后保存模型参数
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))



if __name__ == "__main__":
    main()