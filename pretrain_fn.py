"""
监督方法训练电压相角预测模型
"""

# 预训练模型
import pandapower as pp
# import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from env import PowerGridEnv
from agent1_va import NetVm, NetVa, CombinedModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 训练函数
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0 
    num_samples = 0
    for batch_inputs, batch_targets in train_loader:
        # 将数据移动到CUDA设备
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_inputs)  # 不输入上一个时刻的发电机有功出力
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_inputs.size(0)  # 乘以batch size得到总损失
        num_samples += batch_inputs.size(0)
    return total_loss / num_samples if num_samples > 0 else 0  # 返回平均损失

# 测试函数
def evaluate(model, test_loader, criterion, num_buses=None):
    model.eval()
    total_loss = 0 
    num_samples = 0
    vm_loss = 0
    va_loss = 0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            # 将数据移动到CUDA设备
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item() * batch_inputs.size(0)
            num_samples += batch_inputs.size(0)
            
            # 分别计算电压和相角的损失
            if num_buses is not None:
                vm_pred = outputs[:, :num_buses]
                va_pred = outputs[:, num_buses:]
                vm_target = batch_targets[:, :num_buses]
                va_target = batch_targets[:, num_buses:]
                
                vm_loss += criterion(vm_pred, vm_target).item() * batch_inputs.size(0)
                va_loss += criterion(va_pred, va_target).item() * batch_inputs.size(0)
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    avg_vm_loss = vm_loss / num_samples if num_samples > 0 and num_buses is not None else None
    avg_va_loss = va_loss / num_samples if num_samples > 0 and num_buses is not None else None
    
    return avg_loss, avg_vm_loss, avg_va_loss


# 创建数据集类
class PowerGridDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

if __name__ == "__main__":
    import datetime     
    import os
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # 创建保存模型和图片的文件夹
    os.makedirs('saved_model', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)

    load_data = False  
    case_name = "case9"
    data_save_path = 'saved_data/training_data_9.npz'
    
    # 创建saved_data目录（如果不存在）
    os.makedirs('saved_data', exist_ok=True) 

    num_timesteps = 1000   # 24小时 * 12个5分钟间隔
    random_load = True   # 意味着大量的随机场景
    run_pp = False
    
    env = PowerGridEnv(num_timesteps=num_timesteps, case_name=case_name, random_load=random_load, run_pp=run_pp)  

    obs_dim = env.observation_space.shape[0]
    num_buses = len(env.net.bus)
    
    # 创建模型实例 目前是一个全连接的模型
    # NetVa是相角预测，NetVm是电压预测
    model_va = NetVa(input_channels=obs_dim, output_channels=num_buses, hidden_units=128, khidden=np.array([4, 2, 2, 1]))
    model_vm = NetVm(input_channels=obs_dim, output_channels=num_buses, hidden_units=128, khidden=np.array([4, 2, 2, 1]))
    model = CombinedModel(model_va, model_vm).to(device)

    # 加载模型 
    train = True 

    if not load_data:
        # 收集训练数据
        print("正在收集训练数据...")
        train_inputs = []
        train_targets = []

        # 收集多个episode的数据
        num_episodes = 1
        steps_per_episode = num_timesteps  # 24小时 * 12个5分钟间隔
        
        # 创建总进度条
        total_steps = num_episodes * steps_per_episode
        progress_bar = tqdm(total=total_steps, desc="数据收集进度")

        for episode in range(num_episodes):
            obs = env.reset()   
            while not env.done:
                try:
                    pp.runopp(env.net, verbose=False)  # 运行最优潮流
                    progress_bar.update(1)  # 更新进度条 
                    train_inputs.append(obs)  # 观测的数据包括load的有功和无功
                    
                    # 收集目标值(最优电压和相角)
                    vm_pu = env.net.res_bus.vm_pu.values
                    va_deg = env.net.res_bus.va_degree.values
                    target = np.concatenate([vm_pu, va_deg])
                    train_targets.append(target)
                    
                    # 更新环境状态，获取新的负荷数据  
                    dummy_pg = np.zeros(len(env.net.bus))
                    # 将发电机节点的功率值填入对应位置
                    for i, gen in enumerate(env.net.gen.itertuples()):
                        bus_idx = gen.bus
                        dummy_pg[bus_idx] = env.net.res_gen.p_mw.values[i]
                    action = np.concatenate([vm_pu, dummy_pg])
                    obs, _, _, _ = env.step(action)  
                    print(f"第{episode}个episode的第{env.current_step}步")
                except:
                    print(f"报错，当前场景是可行的，跳过")
                    # 跳过当前时间步    
                    if env.current_step != num_timesteps - 1:
                        env.current_step += 1   # 更新时间步 
                        env.update_load_profiles()    # 更新当前时间步的负荷值 
                        obs = env._get_observation()   # step+1的负荷值
                    else:
                        env.done = True
                    continue
                
        print(f"收集到 {len(train_inputs)} 个有效数据点")

        # 保存训练数据
        np.savez(data_save_path,
                train_inputs=train_inputs,
                train_targets=train_targets)
        print(f"训练数据已保存至: {data_save_path}")
    else:
        # 加载训练数据
        data = np.load(data_save_path)
        train_inputs = data['train_inputs']
        train_targets = data['train_targets']
 
    train_inputs = np.array(train_inputs)   # P_d and Q_d   
    train_targets = np.array(train_targets) # Vm and Va
    # 数据预处理
    # 电压进行归一化，到0-1之间； 相角进行归一化，到-pi-pi之间
    train_targets[:, :num_buses] = (train_targets[:, :num_buses] - env.voltage_low) / (env.voltage_high - env.voltage_low)
    train_targets[:, num_buses:] = train_targets[:, num_buses:] * np.pi / 180  # 将相角转换为弧度
    
    # 将电压和相角进行缩放
    scale_vm = 10.0 # scaling of output Vm
    scale_va = 10.0 # scaling of output Va
    train_targets[:, :num_buses] = train_targets[:, :num_buses] * scale_vm
    train_targets[:, num_buses:] = train_targets[:, num_buses:] * scale_va

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        train_inputs, train_targets, test_size=0.2, random_state=42
    )

    # 创建训练集和测试集的数据加载器
    batch_size = 50
    train_dataset = PowerGridDataset(X_train, y_train)
    test_dataset = PowerGridDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()

    # 创建保存图表的文件夹
    plot_dir = f'training_plots'
    os.makedirs(plot_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if train:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # 训练和评估
        model_save_path = os.path.join('saved_model', f'best_model_{current_time}.pth')
        print("开始训练...")
        num_epochs = 1000
        best_test_loss = float('inf')
        
        # 创建记录训练过程的列表
        train_losses = []
        test_losses = []
                                
        for epoch in tqdm(range(num_epochs), desc="训练进度"):
            # 训练
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            
            # 测试
            test_loss, test_vm_loss, test_va_loss = evaluate(model, test_loader, criterion, num_buses)
            
            # 记录损失
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"保存最佳模型至: {model_save_path}")
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"训练损失: {train_loss:.6f}")
                print(f"测试损失: {test_loss:.6f}")
                if test_vm_loss is not None and test_va_loss is not None:
                    print(f"  电压损失: {test_vm_loss:.6f}, 相角损失: {test_va_loss:.6f}")

        print("训练完成!")
        
        # 绘制损失曲线 
        plt.figure(figsize=(12, 8))
        epochs = range(1, num_epochs + 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Total Loss')
        plt.plot(epochs, test_losses, 'r-', label='Test Total Loss')
        plt.title('Loss Changes During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{plot_dir}/total_loss_{current_time}.png')
        plt.close()
    
    # 加载最佳模型进行最终测试
    model.load_state_dict(torch.load(model_save_path))
    final_test_loss, final_vm_loss, final_va_loss = evaluate(model, test_loader, criterion, num_buses)
    print("\n最终测试结果:")
    print(f"总损失: {final_test_loss:.6f}")
    if final_vm_loss is not None and final_va_loss is not None:
        print(f"电压损失: {final_vm_loss:.6f}")
        print(f"相角损失: {final_va_loss:.6f}")
    
    # 验证模型输出范围
    model.eval()
    with torch.no_grad():
        sample_input = torch.FloatTensor(X_test[:10]).to(device)
        sample_output = model(sample_input)
        vm_output = sample_output[:, :num_buses].cpu().numpy()
        va_output = sample_output[:, num_buses:].cpu().numpy()
        
        print(f"\n模型输出范围验证:")
        print(f"电压输出范围: [{vm_output.min():.4f}, {vm_output.max():.4f}] (期望: [0, 10])")
        print(f"相角输出范围: [{va_output.min():.4f}, {va_output.max():.4f}] (期望: [-31.4, 31.4])")
        
        # 检查输出是否在合理范围内
        vm_in_range = (vm_output >= 0).all() and (vm_output <= 10).all()
        va_in_range = (va_output >= -31.5).all() and (va_output <= 31.5).all()
        
        if vm_in_range and va_in_range:
            print("✓ 模型输出在合理范围内")
        else:
            print("⚠ 警告: 模型输出超出预期范围!")
            if not vm_in_range:
                print(f"  电压输出超出范围: min={vm_output.min():.4f}, max={vm_output.max():.4f}")
            if not va_in_range:
                print(f"  相角输出超出范围: min={va_output.min():.4f}, max={va_output.max():.4f}")

    # 打印保存的模型名称
    print(f"\n保存的模型文件名: {model_save_path}")
    print(f"模型保存路径: {os.path.abspath(model_save_path)}")