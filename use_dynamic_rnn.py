#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ use_dynamic_rnn.py
 Author @ huangjunheng
 Create date @ 2018-06-23 16:23:27
 Description @ 
"""
import torch
import torch.nn as nn

from config import Config
import sequence_data
from sequence_data import SequenceData


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, seq_len):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        last_step_index_list = (seq_len - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1)
        hidden_outputs = out.gather(1, last_step_index_list).squeeze()

        out = self.fc(hidden_outputs)

        return out


class UseDynamicRNN(object):
    """
        数据处理
        """

    def __init__(self):
        self.config = Config()
        # Hyper-parameters

        self.max_seq_len, input_size, num_classes = sequence_data.cal_model_para(self.config.training_file)

        self.model = RNN(input_size, self.config.num_hidden,
                         self.config.num_layers, num_classes).to(device)

        self.loss_and_optimizer()

    def loss_and_optimizer(self):
        """
        Loss and optimizer
        :return: 
        """
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train(self):
        """
        train
        :return: 
        """
        print 'Start training model.'
        training_set = SequenceData(filename=self.config.training_file, max_seq_len=self.max_seq_len)
        for i in range(self.config.training_steps):
            batch_x, batch_y, batch_seqlen = training_set.next(self.config.batch_size)

            batch_x = batch_x.to(device)
            _, batch_y = torch.max(batch_y, 1) # 元组第一个维度为最大值，第二个维度为最大值的索引
            batch_y = batch_y.to(device)
            batch_seqlen = batch_seqlen.to(device)

            # Forward pass
            outputs = self.model(batch_x, batch_seqlen)
            loss = self.criterion(outputs, batch_y)

            # Backward and optimize
            self.optimizer.zero_grad()  # 清空梯度缓存
            loss.backward()  # 反向传播，计算梯度
            self.optimizer.step()  # 利用梯度更新模型参数

            if (i + 1) % 100 == 0:
                print 'Step [{}/{}], Loss: {:.4f}'\
                    .format(i + 1, self.config.training_steps, loss.item())

        # Save the model checkpoint
        print 'Start saving model to "%s".' % self.config.save_model_path
        torch.save(self.model.state_dict(), self.config.save_model_path)

    def test(self, load_model=False):
        """
        test
        :param load_model: 
        :return: 
        """
        if load_model:
            print 'Start loading model from "%s"' % self.config.load_model_path
            self.model.load_state_dict(torch.load(self.config.load_model_path))

        test_set = SequenceData(filename=self.config.test_file, max_seq_len=self.max_seq_len)

        with torch.no_grad():
            correct = 0
            total = 0
            features, labels, seqlen = test_set.get_all_data()

            features = features.to(device)
            _, labels = torch.max(labels, 1)
            labels = labels.to(device)
            seqlen = seqlen.to(device)

            outputs = self.model(features, seqlen)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print 'Test Accuracy of the model: {} %'.format(100 * correct / total)

    def main(self):
        """
        main
        :return: 
        """
        self.train()
        self.test(load_model=True)

if __name__ == '__main__':
    rnn = UseDynamicRNN()
    rnn.main()



