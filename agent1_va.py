"""
神经网络用于预测电压相角
"""

import torch
import torch.nn as nn
import numpy as np 

#%% 
class VoltagePredictor(nn.Module):
    def __init__(self, input_dim, num_buses, voltage_low, voltage_high, angle_low, angle_high):
        super(VoltagePredictor, self).__init__()
        self.num_buses = num_buses
        self.input_dim = input_dim
        
        # 注册电压范围的缓冲区
        self.register_buffer('voltage_low', torch.tensor(voltage_low, dtype=torch.float32))
        self.register_buffer('voltage_high', torch.tensor(voltage_high, dtype=torch.float32))
        self.register_buffer('voltage_range', self.voltage_high - self.voltage_low)

        # 注册相角范围的缓冲区
        self.register_buffer('angle_low', torch.tensor(angle_low, dtype=torch.float32))
        self.register_buffer('angle_high', torch.tensor(angle_high, dtype=torch.float32))
        self.register_buffer('angle_range', self.angle_high - self.angle_low)

        # 定义网络结构
        # 电压预测网络 - 更深层次结构
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2), 
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2), 
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )

        # 相角预测网络 - 更深层次结构
        self.net2 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2), 
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2), 
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )
        
        # 输出层分为电压和相角两部分
        self.voltage_out = nn.Linear(128, num_buses)
        self.angle_out = nn.Linear(128, num_buses)

    def forward(self, x):
        features1 = self.net1(x)
        features2 = self.net2(x)
        
        # 预测电压
        voltage = torch.sigmoid(self.voltage_out(features1))
        voltage = voltage * self.voltage_range + self.voltage_low
        
        # 相角预测（使用tanh缩放至指定范围）
        # angle = torch.tanh(self.angle_out(features)) * 0.5 * self.angle_range + (self.angle_high + self.angle_low) * 0.5
        angle = torch.tanh(self.angle_out(features2)) * 0.5 * self.angle_range + (self.angle_high + self.angle_low) * 0.5

        return torch.cat([voltage, angle], dim=1)
#%%
import torch.nn.functional as F


## NN function
class NetVa(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_units,khidden):
        super(NetVa, self).__init__()
        #''' 
        self.num_layer = khidden.shape[0]
        
        self.fc1 = nn.Linear(input_channels, khidden[0]*hidden_units)  
        if self.num_layer >= 2: 
            self.fc2 = nn.Linear(khidden[0]*hidden_units, khidden[1]*hidden_units)
        
        if self.num_layer >= 3:
            self.fc3 = nn.Linear(khidden[1]*hidden_units, khidden[2]*hidden_units)
        
        if self.num_layer >= 4:
            self.fc4 = nn.Linear(khidden[2]*hidden_units, khidden[3]*hidden_units)
            
        if self.num_layer >= 5:
            self.fc5 = nn.Linear(khidden[3]*hidden_units, khidden[4]*hidden_units)
            
        if self.num_layer >= 6:
            self.fc6 = nn.Linear(khidden[4]*hidden_units, khidden[5]*hidden_units)
            
            
        self.fcbfend = nn.Linear(khidden[khidden.shape[0]-1]*hidden_units, output_channels)   
        self.fcend = nn.Linear(output_channels, output_channels)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        if self.num_layer >= 2:
            x = F.relu(self.fc2(x))
            
        if self.num_layer >= 3:
            x = F.relu(self.fc3(x))
            
        if self.num_layer >= 4:
            x = F.relu(self.fc4(x))
        
        if self.num_layer >= 5:
            x = F.relu(self.fc5(x))
            
        if self.num_layer >= 6:
            x = F.relu(self.fc6(x))
        
        # fixed final two layers
        x = F.relu(self.fcbfend(x))
        x_PredVa = self.fcend(x)
        
        # 相角输出：使用tanh限制到[-1, 1]，然后缩放到训练目标范围
        # 训练目标范围：相角弧度 * 10，大约[-31.4, 31.4]
        scale_va = 10.0
        x_PredVa = torch.tanh(x_PredVa) * (np.pi * scale_va)  # 限制到[-31.4, 31.4]
                
        return x_PredVa
        

class NetVm(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_units,khidden):
        super(NetVm, self).__init__()
        self.num_layer = khidden.shape[0]
        self.fc1 = nn.Linear(input_channels, khidden[0]*hidden_units)
        if self.num_layer >= 2: 
            self.fc2 = nn.Linear(khidden[0]*hidden_units, khidden[1]*hidden_units)
        
        if self.num_layer >= 3:
            self.fc3 = nn.Linear(khidden[1]*hidden_units, khidden[2]*hidden_units)
        
        if self.num_layer >= 4:
            self.fc4 = nn.Linear(khidden[2]*hidden_units, khidden[3]*hidden_units)
            
        if self.num_layer >= 5:
            self.fc5 = nn.Linear(khidden[3]*hidden_units, khidden[4]*hidden_units)
            
        if self.num_layer >= 6:
            self.fc6 = nn.Linear(khidden[4]*hidden_units, khidden[5]*hidden_units)
            
            
        self.fcbfend = nn.Linear(khidden[khidden.shape[0]-1]*hidden_units, output_channels)   
        self.fcend = nn.Linear(output_channels, output_channels) 
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        if self.num_layer >= 2:
            x = F.relu(self.fc2(x))
            
        if self.num_layer >= 3:
            x = F.relu(self.fc3(x))
            
        if self.num_layer >= 4:
            x = F.relu(self.fc4(x))
        
        if self.num_layer >= 5:
            x = F.relu(self.fc5(x))
            
        if self.num_layer >= 6:
            x = F.relu(self.fc6(x))
        
        # fixed final two layers
        x = F.relu(self.fcbfend(x))
        x_PredVm = self.fcend(x)
        
        # 电压输出：使用sigmoid限制到[0, 1]，然后缩放到训练目标范围[0, 10]
        scale_vm = 10.0
        x_PredVm = torch.sigmoid(x_PredVm) * scale_vm  # 限制到[0, 10]

        return x_PredVm


# 将model_va和model_vm封装成一个组合模型
class CombinedModel(torch.nn.Module):
    def __init__(self, model_va, model_vm):
        super(CombinedModel, self).__init__()
        self.model_va = model_va  # 相角预测模型
        self.model_vm = model_vm  # 电压预测模型
        
    def forward(self, x):
        # 分别通过两个模型获取预测结果
        va_pred = self.model_va(x)
        vm_pred = self.model_vm(x)
        
        # 将电压和相角预测结果拼接在一起
        combined_output = torch.cat([vm_pred, va_pred], dim=1)
        return combined_output