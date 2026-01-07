"""
构建一个简单的电力系统，并使用pandapower进行潮流计算

# 3-13 目前的环境的负荷是生成的，且不需要预测，然后环境更新部分，目前看起来都没啥问题，计算的约束违反和奖励函数，目前看起来也没啥问题
# 但是现在没加入这个相邻时间功率之间的约束，需要加入
# 3-16 目前环境里的负荷曲线是固定的，然后网络拓扑也是不变的，需要修改。
"""


import gym
from gym import spaces
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
import torch
from scipy.ndimage import gaussian_filter1d

class PowerGridEnv(gym.Env):
    """Custom Power Grid Environment that follows the OpenAI Gym interface"""
    
    def __init__(self, num_timesteps=24*12*300, case_name="case9", random_load=False, run_pp=True, consider_renewable_generation=False):
        super(PowerGridEnv, self).__init__()
        
        # Create a pandapower network
        self.case_name = case_name
        self.net = self._create_network()   # init network
        self.num_timesteps = num_timesteps
        self.random_load = random_load
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # 如果不是pretrain，则考虑可再生能源的出力
        self.consider_renewable_generation = consider_renewable_generation

        self.run_pp = run_pp

        pp.runpp(self.net)

        # 计算导纳矩阵
        Ybus, _, _ = makeYbus(self.net["_ppc"]["baseMVA"], self.net["_ppc"]["bus"], self.net["_ppc"]["branch"])
        self.Ybus = Ybus.toarray()  # 转换为numpy数组
        # pp.runpp(self.net)
        # self.Ybus = self.net._ppc["internal"]["Ybus"].todense()  # init Ybus
        
        # Define action and observation spaces  
        self.num_buses = len(self.net.bus)
 
        self.voltage_low = self.net.bus.min_vm_pu.values  # 电压下限
        self.voltage_high = self.net.bus.max_vm_pu.values  # 电压上限
        
        # 相角范围通常在-π到π之间 
        self.angle_low = np.ones(self.num_buses) * (-180)  # 相角下限
        self.angle_high = np.ones(self.num_buses) * 180  # 相角上限 
        
        # 合并电压和相角的动作空间
        action_low = np.concatenate([self.voltage_low, self.angle_low])
        action_high = np.concatenate([self.voltage_high, self.angle_high])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        
        # Example: observe bus voltages, line loadings, and generator outputs
        p_mw_dim = len(self.net.load['p_mw'])  # 负荷的有功功率
        q_mvar_dim = len(self.net.load['q_mvar'])  # 负荷的无功功率 
        # p_mw_dim_gen = len(self.net.gen['p_mw'])  # 发电机的有功功率
        obs_dim = p_mw_dim + q_mvar_dim  #  + p_mw_dim_gen
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                          shape=(obs_dim,), dtype=np.float32)
        
        # 生成负荷功率曲线   
        if self.random_load:
            self._generate_random_load_profiles() 
        else:
            # 随机生成负荷功率曲线
            self._generate_load_profiles()
            # 生成风电出力曲线（仅当case为case118时）
            if self.case_name == "case118" and self.consider_renewable_generation:
                self._generate_wind_profiles()

        # 将每个发电机的功率爬坡速率限制在其最大出力的1/20
        # self.threshold = {}
        # if len(self.net.gen) > 0:
        #     for gen in self.net.gen.itertuples():
        #         max_p = gen.max_p_mw
        #         self.threshold[gen.Index] = max_p / 20
        # else:
        #     self.threshold = 10  # 默认值


    def draw_load_profiles(self):
        """绘制总负荷曲线和负荷比率曲线"""
        import matplotlib
        # 在导入 pyplot 之前设置后端为 Agg
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 清除之前的图形
        plt.close('all')
        
        num_loads = len(self.net.load)
        num_timesteps = len(self.load_profiles_p[0])
        
        # 创建时间轴标签(小时)
        hours = np.linspace(0, 24, num_timesteps)
        
        # 计算总有功功率
        total_active_power = np.sum(self.load_profiles_p, axis=0)
        # 计算比率（相对于平均值）
        ratio = total_active_power / np.mean(total_active_power)
        
        # 创建图形和子图
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 绘制总有功功率曲线
        ax1.plot(hours, total_active_power, 'b-', label='Active load')
        ax1.set_xlabel('Time (hour)')
        ax1.set_ylabel('Active load (MW)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xlim(0, 24)
        # 设置时间刻度为3小时间隔
        ax1.set_xticks(np.arange(0, 25, 3))
        ax1.set_xticklabels([f'{h:d}:00' for h in range(0, 25, 3)])
        ax1.grid(True)
        
        # 创建第二个Y轴
        ax2 = ax1.twinx()
        ax2.plot(hours, ratio, 'r-', label='Ratio')
        ax2.set_ylabel('Ratio', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('24-hour Load Profile')
        plt.tight_layout()
        
        # 保存图形到文件
        plt.savefig('load_profile_total.png')
        
        # 另外绘制每个负荷节点的曲线
        plt.figure(figsize=(12, 6))
        for i in range(num_loads):
            plt.plot(hours, self.load_profiles_p[i], label=f'Load {i}')
        
        plt.xlabel('Time (hour)')
        plt.ylabel('Active Power (MW)')
        plt.title('24-hour Load Profiles for Each Node')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # 保存第二个图形到文件
        plt.savefig('load_profiles_individual.png')
        
        # 关闭所有图形
        plt.close('all')

    def _generate_load_profiles(self):
        """
        生成基于真实负荷模式的24小时负荷曲线
        具有明显的早晚高峰特征
        """
        num_timesteps = 24*12
        # num_timesteps = self.num_timesteps  # 24小时 * 12个5分钟
        num_loads = len(self.net.load)
        
        # 基础负荷值
        base_p = self.net.load['p_mw'].values
        base_q = self.net.load['q_mvar'].values
        
        # 初始化负荷数据数组
        self.load_profiles_p = np.zeros((num_loads, num_timesteps))
        self.load_profiles_q = np.zeros((num_loads, num_timesteps))
        
        # 创建时间轴
        time_hours = np.linspace(0, 24, num_timesteps)
        
        # 设置随机种子以保证一致性
        np.random.seed(42)
        
        # 定义双峰负荷模式的系数曲线（基于图像中的模式）
        load_pattern = np.zeros(num_timesteps)
        
        # 为每个时间点定义负荷系数
        for i, hour in enumerate(time_hours):
            if hour < 3:  # 0:00-3:00 凌晨低谷
                load_pattern[i] = 0.80 + 0.01 * (3 - hour)
            elif hour < 5:  # 3:00-5:00 保持平稳低谷
                load_pattern[i] = 0.80
            elif hour < 9:  # 5:00-9:00 早高峰上升
                load_pattern[i] = 0.80 + 0.20 * ((hour - 5) / 4)
            elif hour < 12:  # 9:00-12:00 早高峰下降
                load_pattern[i] = 1.00 - 0.10 * ((hour - 9) / 3)
            elif hour < 15:  # 12:00-15:00 中午降至低谷
                load_pattern[i] = 0.90 - 0.10 * ((hour - 12) / 3)
            elif hour < 18:  # 15:00-18:00 晚高峰上升
                load_pattern[i] = 0.80 + 0.15 * ((hour - 15) / 3)
            elif hour < 21:  # 18:00-21:00 晚高峰下降
                load_pattern[i] = 0.95 - 0.10 * ((hour - 18) / 3)
            else:  # 21:00-24:00 逐渐回落到低谷
                load_pattern[i] = 0.85 - 0.05 * ((hour - 21) / 3)
        
        # 添加小幅随机波动
        random_variations = 0.03 * np.random.randn(num_timesteps)
        load_pattern = load_pattern + random_variations
        
        # 确保负荷系数在合理范围内
        load_pattern = np.clip(load_pattern, 0.75, 1.0)
        
        # 应用负荷模式到每个负荷节点
        for i in range(num_loads):
            # 添加节点特定的小幅随机性
            node_variations = 0.01 * np.random.randn(num_timesteps)
            node_pattern = load_pattern + node_variations
            
            # 生成有功功率曲线
            self.load_profiles_p[i] = base_p[i] * node_pattern
            
            # 生成无功功率曲线，保持相同的模式
            self.load_profiles_q[i] = base_q[i] * node_pattern
        
        # 确保负荷值为正
        self.load_profiles_p = np.maximum(self.load_profiles_p, 0.1 * base_p[:, np.newaxis])
        self.load_profiles_q = np.maximum(self.load_profiles_q, 0.1 * base_q[:, np.newaxis])
        
        # 将负荷曲线复制300次，生成更长的时间序列 
        if self.num_timesteps > num_timesteps:  # 如果时间步数大于num_timesteps，则将负荷曲线复制，满足num_timesteps长度要求
            self.load_profiles_p = np.tile(self.load_profiles_p, (1, self.num_timesteps // num_timesteps))
            self.load_profiles_q = np.tile(self.load_profiles_q, (1, self.num_timesteps // num_timesteps))
        print(f"self.load_profiles_p.shape: {self.load_profiles_p.shape}")
        
        # 计算并存储活跃负荷和比率数据用于可视化
        self.active_load_total = np.sum(self.load_profiles_p, axis=0)
        self.ratio_data = self.active_load_total / np.mean(self.active_load_total)

        self.draw_load_curve = False
        if self.draw_load_curve:  # debug 绘制负荷曲线
            self.draw_load_profiles()
    
    def _generate_wind_profiles(self):
        """
        生成风电场的出力曲线
        针对IEEE 118-bus system中的节点59、90和116
        基于真实风电场的出力特征图精确匹配
        """
        num_timesteps = 288  # 24小时 * 12个5分钟
        time_hours = np.linspace(0, 24, num_timesteps)
        
        # 初始化风电出力数组
        self.wind_profiles = {
            59: np.zeros(num_timesteps),
            90: np.zeros(num_timesteps),
            116: np.zeros(num_timesteps)
        }
        
        # 设置随机种子以保证一致性
        np.random.seed(42)
        
        # 基础曲线 - Bus 59 (黄线)
        base_59 = np.zeros(num_timesteps)
        for i, hour in enumerate(time_hours):
            if hour < 3:  # 0:00-3:00 初始较高，快速下降
                base_59[i] = 10 - 2 * hour
            elif hour < 12:  # 3:00-12:00 维持低水平
                base_59[i] = 4 
            elif hour < 15:  # 12:00-15:00 快速上升到20MW
                base_59[i] = 4 + (20-4) * (hour - 12) / 3
            elif hour < 17:  # 15:00-17:00 维持在20MW左右
                base_59[i] = 20
            elif hour < 19:  # 17:00-19:00 上升到27MW左右并震荡
                base_59[i] = 20 + (27-20) * (hour - 17) / 2
            elif hour < 21:  # 19:00-21:00 快速下降
                base_59[i] = 27 - 11.5 * (hour - 19)
            else:  # 21:00-24:00 继续下降至4MW左右
                base_59[i] = 4 + 0.5 * np.sin((hour - 21) * np.pi / 3)
        
        # 基础曲线 - Bus 90 (蓝线)
        base_90 = np.zeros(num_timesteps)
        for i, hour in enumerate(time_hours):
            if hour < 1:  # 0:00-1:00 
                base_90[i] = 5 - 3 * hour
            elif hour < 8:  # 1:00-8:00 保持低水平
                base_90[i] = 1.5
            elif hour < 19:  # 8:00-19:00 中等水平，有规律波动
                base_90[i] = 3.5
            else:  # 19:00-24:00 升至高峰
                base_90[i] = 3.5 + 10 * (1 - np.exp(-(hour - 19) / 2))
        
        # 基础曲线 - Bus 116 (绿线) - 修改为9:00后逐步上升到15MW左右
        base_116 = np.zeros(num_timesteps)
        for i, hour in enumerate(time_hours):
            if hour < 2:  # 0:00-2:00 
                base_116[i] = 3.5 - 1.5 * hour
            elif hour < 6:  # 2:00-6:00 
                base_116[i] = 1 + 0.8 * (hour - 2)
            elif hour < 9:  # 6:00-9:00 维持在低水平并有震荡
                base_116[i] = 4 + 0.5 * np.sin((hour - 6) * np.pi)
            elif hour < 21:  # 9:00-21:00 逐步上升到15MW左右
                # 从4MW缓慢上升到15MW，使用非线性曲线使上升速度后期变缓
                progress = (hour - 9) / 12  # 0-1之间的进度值
                curve_factor = np.sqrt(progress)  # 非线性因子，使曲线更平滑
                base_116[i] = 4 + (15 - 4) * curve_factor
            else:  # 21:00-24:00 保持在15MW左右
                base_116[i] = 15 + 0.8 * np.sin((hour - 21) * np.pi / 1.5)
        
        # 添加波动 - 使用更合适的波动模式
        # 生成不同频率的噪声分量，然后组合
        for bus, base in zip([59, 90, 116], [base_59, base_90, base_116]):
            # 初始化波动分量
            oscillation = np.zeros(num_timesteps)
            
            # 添加长周期波动 (约3-4小时)
            num_long_waves = 3
            amp_long = 1.2 if bus == 90 else 1.0  # Bus 90波动更明显
            for j in range(num_long_waves):
                freq = 0.15 + 0.05 * j  # 频率变化
                phase = np.random.rand() * 2 * np.pi
                oscillation += amp_long * np.sin(freq * time_hours + phase)
            
            # 添加中周期波动 (约1-2小时)
            num_mid_waves = 5
            amp_mid = 0.7 if bus == 90 else 0.5
            for j in range(num_mid_waves):
                freq = 0.3 + 0.1 * j
                phase = np.random.rand() * 2 * np.pi
                oscillation += amp_mid * np.sin(freq * time_hours + phase)
            
            # 添加高频小震荡（约15-30分钟周期）
            num_high_freq_waves = 8
            amp_high = 0.3 if bus == 90 else 0.2
            for j in range(num_high_freq_waves):
                freq = 1.0 + 0.3 * j
                phase = np.random.rand() * 2 * np.pi
                oscillation += amp_high * np.sin(freq * time_hours + phase)
                
            # 添加更高频小震荡 (每5-10分钟)
            num_very_high_freq = 5
            amp_very_high = 0.15 if bus == 90 else 0.1
            for j in range(num_very_high_freq):
                freq = 2.5 + 0.8 * j
                phase = np.random.rand() * 2 * np.pi
                oscillation += amp_very_high * np.sin(freq * time_hours + phase)
            
            # 按照时间段调整波动幅度
            oscillation_amplitude = np.ones_like(base)
            
            if bus == 59:  # Bus 59在高峰期波动较大
                for i, hour in enumerate(time_hours):
                    if hour < 12:
                        oscillation_amplitude[i] = 0.5  # 低水平期波动增加
                    elif hour < 15:
                        oscillation_amplitude[i] = 0.6  # 上升期波动适中
                    elif hour < 17:
                        oscillation_amplitude[i] = 0.8  # 20MW平台期波动适中
                    elif hour < 19:
                        oscillation_amplitude[i] = 1.5  # 17:00-19:00上升到27MW时波动最大
                    else:
                        oscillation_amplitude[i] = 0.5  # 下降期波动减小
            
            elif bus == 90:  # Bus 90在全天都有显著波动
                for i, hour in enumerate(time_hours):
                    if hour < 9:
                        oscillation_amplitude[i] = 0.6  # 低水平期波动增加
                    elif hour < 19:
                        oscillation_amplitude[i] = 1.2  # 中间期波动加大
                    else:
                        oscillation_amplitude[i] = 2.0  # 高峰期波动最大
            
            elif bus == 116:  # Bus 116的波动调整
                for i, hour in enumerate(time_hours):
                    if hour < 9:
                        oscillation_amplitude[i] = 0.8  # 初始阶段中等波动
                    elif hour < 15:
                        oscillation_amplitude[i] = 0.9  # 上升初期适中波动
                    elif hour < 21:
                        oscillation_amplitude[i] = 1.2  # 上升后期波动加大
                    else:
                        oscillation_amplitude[i] = 1.5  # 稳定在高位时波动更明显
            
            # 特殊处理 - Bus 90在早上有一段时间接近零
            if bus == 90:
                zero_mask = (time_hours >= 6) & (time_hours <= 8)
                base[zero_mask] = 0.2
                oscillation_amplitude[zero_mask] = 0.1
            
            # 特殊处理 - Bus 116在8-9小时处有一个显著下降
            if bus == 116:
                dip_mask = (time_hours >= 8) & (time_hours <= 9)
                base[dip_mask] = 3.0  # 调整为轻微下降，但不会太低
                oscillation_amplitude[dip_mask] = 0.5
            
            # 特殊处理 - Bus 59在17:00-19:00时段有更强的震荡
            if bus == 59:
                strong_osc_mask = (time_hours >= 17) & (time_hours <= 19)
                # 添加额外的震荡分量
                extra_osc = 1.2 * np.sin(6 * time_hours[strong_osc_mask] + np.random.rand() * np.pi)
                self.wind_profiles[bus][strong_osc_mask] = base[strong_osc_mask]
                # 应用基础波动
                self.wind_profiles[bus] = base + oscillation * oscillation_amplitude
                # 叠加额外震荡
                self.wind_profiles[bus][strong_osc_mask] += extra_osc
            else:
                # 将基础曲线和波动组合
                self.wind_profiles[bus] = base + oscillation * oscillation_amplitude
            
            # 确保出力非负
            self.wind_profiles[bus] = np.maximum(self.wind_profiles[bus], 0)
            
            # 添加小的随机抖动，使曲线更符合实际
            jitter = 0.2 * np.random.randn(num_timesteps)
            self.wind_profiles[bus] += jitter
            
            # 轻微平滑处理，只是去除过于尖锐的波动
            self.wind_profiles[bus] = gaussian_filter1d(self.wind_profiles[bus], sigma=0.3)
        
        # 确保所有曲线都在0-30范围内
        for bus in [59, 90, 116]:
            self.wind_profiles[bus] = np.clip(self.wind_profiles[bus], 0, 30)

        
        # 将负荷曲线复制300次，生成更长的时间序列   
        if self.num_timesteps > num_timesteps:  # 如果时间步数大于num_timesteps，则将负荷曲线复制，满足num_timesteps长度要求
            for bus in [59, 90, 116]:
                self.wind_profiles[bus] = np.tile(self.wind_profiles[bus], (self.num_timesteps // num_timesteps)) 
        
        # 添加绘制风电出力曲线的功能
        self.draw_wind_curve = False
        if self.draw_wind_curve:
            self.draw_wind_profiles()
        
    def draw_wind_profiles(self):
        """绘制风电场的出力曲线，风格与原始图片一致"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 清除之前的图形
        plt.close('all')
        
        num_timesteps = len(self.wind_profiles[59])
        hours = np.linspace(0, 24, num_timesteps)
        
        # 设置图表样式以匹配原始图像
        plt.figure(figsize=(12, 8))
        
        # 使用与原图相似的颜色
        plt.plot(hours, self.wind_profiles[59], color='#FFA500', label='Bus 59', linewidth=1.5)
        plt.plot(hours, self.wind_profiles[90], color='#1E90FF', label='Bus 90', linewidth=1.5)
        plt.plot(hours, self.wind_profiles[116], color='#228B22', label='Bus 116', linewidth=1.5)
        
        # 设置轴标签和标题
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Wind generation (MW)', fontsize=20)
        plt.title('24-hour Wind Power Generation Profile', fontsize=16)
        
        # 设置网格
        plt.grid(True, linestyle='-', alpha=0.7)
        
        # 设置刻度
        plt.yticks(np.arange(0, 31, 5), fontsize=14)
        plt.xticks(np.arange(0, 25, 3), [f'{h:d}:00' for h in range(0, 25, 3)], fontsize=14)
        
        # 设置y轴范围，与原图一致
        plt.ylim(0, 30)
        plt.xlim(0, 24)
        
        # 添加图例
        plt.legend(fontsize=16, loc='upper left')
        
        # 添加边框
        ax = plt.gca()
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        plt.tight_layout()
        
        # 保存图形到文件
        plt.savefig('wind_profiles.png', dpi=300)
        plt.close()

    def _create_network(self):
        """Create a pandapower network for the environment"""
        # net = pp.create_empty_network()
        # ... create your network components ...
        if self.case_name == "case5":
            net = pn.case5()
        elif self.case_name == "case9":
            net = pn.case9()
        elif self.case_name == "case30":
            net = pn.case30()
        elif self.case_name == "case57":
            net = pn.case57()
        elif self.case_name == "case118":
            net = pn.case118()
        print(f"\n {self.case_name} created")
        return net
    
    def _get_observation(self):
        """Convert pandapower network state to observation vector"""
        # pp.runpp(self.net)
        # 提取相关信息作为观测
        load_p = self.net.load['p_mw'].values  # 负荷的有功功率
        load_q = self.net.load['q_mvar'].values  # 负荷的无功功率
        # gen_p = self.net.gen['p_mw'].values
        
        # observation = np.concatenate([load_p, load_q, gen_p])
        observation = np.concatenate([load_p, load_q])
        return observation 
    
    def update_load_profiles(self):
        """Update the load profiles for the next time step"""
        self.net.load['p_mw'] = self.load_profiles_p[:, self.current_step]
        self.net.load['q_mvar'] = self.load_profiles_q[:, self.current_step] 

        """更新风电出力曲线，并调整相应节点的负荷"""
        if self.case_name == "case118" and self.consider_renewable_generation:
            # 更新风电出力
            print(f"更新风电出力，当前时间步为{self.current_step}")
            wind_buses = [59, 90, 116]
            for bus in wind_buses:
                wind_power = self.wind_profiles[bus][self.current_step]
                
                # 查找该节点是否有负荷
                load_indices = self.net.load[self.net.load.bus == bus].index
                if len(load_indices) > 0:
                    # 如果节点有负荷，则调整负荷值（减去风电出力）
                    load_idx = load_indices[0]
                    original_load = self.load_profiles_p[load_idx][self.current_step]
                    
                    # 负荷等于总负荷减去新能源出力
                    adjusted_load = max(0.1, original_load - wind_power)  # 确保负荷不小于0.1
                    self.net.load.at[load_idx, 'p_mw'] = adjusted_load
                    
                    # 按相同比例调整无功功率
                    # if original_load > 0:
                    #     ratio = adjusted_load / original_load
                    #     original_q = self.load_profiles_q[load_idx][self.current_step]
                    #     self.net.load.at[load_idx, 'q_mvar'] = original_q * ratio

    def _calculate_reward(self):
        """Calculate reward based on network state and physical constraints"""
        if not self.converged:
            return -100  # Large penalty for non-convergence
        
        # 1. 发电成本最小化 (假设二次函数形式的发电成本)
        gen_costs = 0
        for i, gen in enumerate(self.net.gen.itertuples()):
            p_g = gen.p_mw
            # 使用pandapower案例中的多项式成本系数
            if hasattr(self.net, 'poly_cost') and len(self.net.poly_cost) > 0:
                # 查找对应发电机的成本系数
                gen_cost_data = self.net.poly_cost[(self.net.poly_cost.et == 'gen') & 
                                                  (self.net.poly_cost.element == i)]
                if not gen_cost_data.empty:
                    # 使用二次多项式成本: cp2*p^2 + cp1*p + cp0
                    cp2 = gen_cost_data.cp2_eur_per_mw2.values[0] if 'cp2_eur_per_mw2' in gen_cost_data else 0
                    cp1 = gen_cost_data.cp1_eur_per_mw.values[0] if 'cp1_eur_per_mw' in gen_cost_data else 0
                    cp0 = gen_cost_data.cp0_eur.values[0] if 'cp0_eur' in gen_cost_data else 0
                    gen_cost = cp2 * p_g**2 + cp1 * p_g + cp0
                else:
                    # 默认成本系数
                    gen_cost = 0.1 * p_g**2 + 20 * p_g + 100
                    print(f"发电机 {i} 的成本系数是手动设置的")
            else:
                # 默认成本系数
                gen_cost = 0.1 * p_g**2 + 20 * p_g + 100
                print(f"发电机 {i} 的成本系数是手动设置的")
            
            gen_costs += gen_cost
        
        # 2，3. 发电机有功、无功功率限制违反惩罚
        p_violation = 0
        q_violation = 0
        for i, gen in enumerate(self.net.gen.itertuples()):
            p_min = gen.min_p_mw if hasattr(gen, 'min_p_mw') else 0
            p_max = gen.max_p_mw if hasattr(gen, 'max_p_mw') else float('inf')
            p_g = gen.p_mw
            p_violation += max(0, p_min - p_g) + max(0, p_g - p_max)

            # 直接从res_gen获取发电机的无功功率
            q_g = self.net.res_gen.q_mvar.iloc[i]
            q_min = gen.min_q_mvar if hasattr(gen, 'min_q_mvar') else 0
            q_max = gen.max_q_mvar if hasattr(gen, 'max_q_mvar') else float('inf')
            q_violation += max(0, q_min - q_g) + max(0, q_g - q_max)
        
        # 4. 所有节点电压限制违反惩罚
        v_violation = 0 
        for i, bus in enumerate(self.net.res_bus.itertuples()):
            v_pu = bus.vm_pu
            v_violation += max(0, self.voltage_low[i] - v_pu) + max(0, v_pu - self.voltage_high[i])
        
        # 5. 所有支路电流限制违反惩罚
        i_violation = 0
        if hasattr(self.net, 'res_line'):
            for line in self.net.res_line.itertuples():
                i_ka = abs(line.i_ka)  # 取电流绝对值
                # 从原始line数据中获取最大电流限制
                line_idx = line.Index
                i_max = self.net.line.max_i_ka.iloc[line_idx]
                i_violation += max(0, i_ka - i_max)
        
        # 加权惩罚项
        # w1,w2,w3 单位为 MW^-1, w4(αv)单位为 p.u.^-1
        w1, w2, w3 = 1.0, 1.0, 1.0  # 有功/无功/电压违反惩罚系数 (1/MW)
        w4 = 100.0  # 电流违反惩罚系数 αv (1/p.u.)
        total_penalty = (w1 * p_violation +  # MW * (1/MW) = 1
                        w2 * q_violation +   # Mvar * (1/MW) = 1  
                        w3 * v_violation +   # p.u. * (1/MW) = p.u./MW
                        w4 * i_violation)    # p.u. * (1/p.u.) = 1
        
        # 最终奖励 = 负的发电成本 - 惩罚项
        a = 10**-4
        reward = -a * gen_costs - total_penalty
        # 记录约束违反程度
        constraint_violations = {
            'p_violation': p_violation,
            'q_violation': q_violation,
            'v_violation': v_violation,
            'i_violation': i_violation,
            'total_penalty': total_penalty,
            'gen_costs': gen_costs
        }
        
        # 将约束违反信息保存到环境中，以便外部访问
        self.constraint_violations = constraint_violations
        return reward
    
    def print_bus_info(self, net): # 定义一个函数来打印节点信息
        print("\n节点信息:")
        for i, bus in enumerate(net.bus.itertuples()):
            bus_type = "未知类型"
            
            # 检查是否为发电机节点
            if i in net.gen.bus.values:
                bus_type = "发电机节点"
                gen_indices = np.where(net.gen.bus.values == i)[0]
                gen_info = [f"发电机 {idx}：{net.gen.iloc[idx]['p_mw']} MW" for idx in gen_indices]
                bus_type += f" ({', '.join(gen_info)})"
            
            # 检查是否为负荷节点
            if i in net.load.bus.values:
                if "发电机节点" in bus_type:
                    bus_type = "PV节点 (发电机+负荷)"
                else:
                    bus_type = "PQ节点 (负荷)"
                load_indices = np.where(net.load.bus.values == i)[0]
                load_info = [f"负荷 {idx}：{net.load.iloc[idx]['p_mw']} MW" for idx in load_indices]
                bus_type += f" ({', '.join(load_info)})"
            
            # 检查是否为平衡节点
            if hasattr(net, 'ext_grid') and i in net.ext_grid.bus.values:
                bus_type = "平衡节点 (Slack Bus)"
            
            print(f"节点 {i}: {bus_type}, 名称: {bus.name if hasattr(bus, 'name') else '未命名'}")
        print("\n")
 
    def reset(self):
        """Reset the environment to an initial state""" 
        # self.net = self._create_network()
        self.done = False
        
        # 调用函数打印节点信息
        # self.print_bus_info(self.net)
        if self.random_load:
            self._generate_random_load_profiles()
        
        # 初始化时间步
        self.current_step = 0
        
        # 设置初始负荷
        self.net.load['p_mw'] = self.load_profiles_p[:, 0]   
        self.net.load['q_mvar'] = self.load_profiles_q[:, 0]

        # 如果是case118，更新负荷以考虑可再生能源的出力
        if 'case118' in self.case_name and self.consider_renewable_generation:
            # 检查是否有风电场节点
            wind_farm_buses = [59, 90, 116]
            for bus in wind_farm_buses:
                if bus in self.net.load.bus.values:
                    # 找到对应的负荷索引
                    load_idx = self.net.load[self.net.load.bus == bus].index[0]
                    # 获取当前时间步的风电出力
                    wind_power = self.wind_profiles[bus][0]  # 初始时间步为0
                    # 更新负荷值 = 原始负荷 - 风电出力
                    original_load = self.load_profiles_p[load_idx, 0]
                    # 确保负荷减去风电后不会变为负值
                    new_load = max(0.1, original_load - wind_power)
                    # 更新负荷值
                    self.net.load.at[load_idx, 'p_mw'] = new_load 
        
        # 设置初始发电机功率
        # 方法1：根据负荷总和按比例分配
        if hasattr(self.net, 'gen') and len(self.net.gen) > 0:
            total_load = np.sum(self.net.load['p_mw'].values)
            num_gens = len(self.net.gen)
            # 考虑一些损耗，总发电量略大于总负荷
            total_gen = total_load * 1.05
            # 按比例分配给各个发电机
            self.net.gen['p_mw'] = np.ones(num_gens) * (total_gen / num_gens)
        
        # Get initial observation
        observation = self._get_observation()  # reset
        # print(f"初始观察: {observation}")
        return observation
    
    def step(self, action):  
        """Execute one time step within the environment"""  
        # update the active power and voltage of the generator nodes P_g, V_g
        v_g = action[:self.num_buses]
        p_g = action[self.num_buses:]
        # 直接遍历发电机表，更新每个发电机的设置值
        for gen in self.net.gen.itertuples():
            bus_idx = gen.bus  # 获取发电机连接的母线索引
            gen_idx = gen.Index  # 获取发电机在gen表中的索引
            
            # 直接使用母线索引从动作中获取对应的设定值
            self.net.gen.at[gen_idx, 'p_mw'] = p_g[bus_idx]
            self.net.gen.at[gen_idx, 'vm_pu'] = v_g[bus_idx]
        
        # Calculate reward
        if self.run_pp:
            try:
                pp.runpp(self.net)   # run the power flow calculation by the newton-raphson method
                self.converged = True
            except:
                self.converged = False
                print("潮流计算不收敛") 

            reward = self._calculate_reward()
        else:
            reward = None
            self.converged = True

        self.current_step += 1 # 更新时间步

        # Get new observation
        if self.current_step < self.num_timesteps: 
            self.update_load_profiles()             # update the load value of the current time step
            observation = self._get_observation()   # step 
        else:
            observation = self._get_observation()   # step 

        # Check if episode is done
        if not self.converged:
            self.done = True
        if self.current_step >= self.num_timesteps:
            self.done = True
        
        # Additional info
        info = {}
        
        return observation, reward, self.done, info
    
    def step_pre(self, action):
        """Execute one time step within the environment"""  
        # update the active power and voltage of the generator nodes P_g, V_g
        v_g = action[:self.num_buses]
        p_g = action[self.num_buses:]
        # 直接遍历发电机表，更新每个发电机的设置值
        for gen in self.net.gen.itertuples():
            bus_idx = gen.bus  # 获取发电机连接的母线索引
            gen_idx = gen.Index  # 获取发电机在gen表中的索引
            
            # 直接使用母线索引从动作中获取对应的设定值
            self.net.gen.at[gen_idx, 'p_mw'] = p_g[bus_idx]
            self.net.gen.at[gen_idx, 'vm_pu'] = v_g[bus_idx]
        
        # Calculate reward 
        try:
            pp.runpp(self.net)   # run the power flow calculation by the newton-raphson method
            self.converged = True
        except:
            self.converged = False
            print("潮流计算不收敛") 

        reward = self._calculate_reward()
        
        # Get observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = False
        if not self.converged:
            done = True
        
        info = {}
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            # Print key network statistics
            print(f"Bus Voltages: {self.net.res_bus.vm_pu.values}")
            print(f"Line Loadings: {self.net.res_line.loading_percent.values}")
        return

    def update_gen_constraints(self):
        for gen in self.net.gen.itertuples():
            previous_p = self.net.res_gen.p_mw.values[gen.Index]
            min_p = max(0, previous_p - self.threshold[gen.Index])
            max_p = previous_p + self.threshold[gen.Index]
            self.net.gen.loc[gen.Index, 'min_p_mw'] = min_p
            self.net.gen.loc[gen.Index, 'max_p_mw'] = max_p

    def _generate_random_load_profiles(self):
        """
        生成随机负荷曲线，服从默认负荷正负10%范围内的均匀分布
        
        参数:
        num_timesteps (int): 负荷曲线的时间步长，默认为288（24小时×12个5分钟）
        """
        num_loads = len(self.net.load)
        num_timesteps = self.num_timesteps
        
        # 基础负荷值
        base_p = self.net.load['p_mw'].values
        base_q = self.net.load['q_mvar'].values
        
        # 初始化负荷数据数组
        self.load_profiles_p = np.zeros((num_loads, num_timesteps))
        self.load_profiles_q = np.zeros((num_loads, num_timesteps))
        
        # 设置随机种子以保证一致性
        # np.random.seed(42)    # todo: 不随机了，也就是初始化的时候负荷是不固定的
        
        # 为每个负荷节点生成随机负荷曲线
        for i in range(num_loads):
            # 生成服从均匀分布的随机系数，范围为0.9到1.1（即默认负荷的正负10%）
            random_factors_p = np.random.uniform(0.9, 1.1, num_timesteps)
            random_factors_q = np.random.uniform(0.9, 1.1, num_timesteps)
            
            # 应用随机系数到基础负荷值
            self.load_profiles_p[i] = base_p[i] * random_factors_p
            self.load_profiles_q[i] = base_q[i] * random_factors_q
        
        # 确保负荷值为正
        self.load_profiles_p = np.maximum(self.load_profiles_p, 0.1 * base_p[:, np.newaxis])
        self.load_profiles_q = np.maximum(self.load_profiles_q, 0.1 * base_q[:, np.newaxis])
        
        # 计算并存储活跃负荷和比率数据用于可视化
        # self.active_load_total = np.sum(self.load_profiles_p, axis=0)
        # self.ratio_data = self.active_load_total / np.mean(self.active_load_total)

        # 对于调试目的，如果需要可以绘制负荷曲线
        self.draw_load_curve = False
        if self.draw_load_curve:
            self.draw_load_profiles()



if __name__ == "__main__":
    # 创建环境实例
    case_name = "case118"   
    env = PowerGridEnv(case_name=case_name, consider_renewable_generation=False)
    
    # 重置环境
    obs = env.reset()
    print("初始观测值:", obs)
    
    # 创建一个简单的动作
    num_buses = len(env.net.bus)
    v_g = np.ones(num_buses)  # 所有节点电压设为1.0标幺值
    p_g = np.zeros(num_buses)  # 创建发电机出力矩阵
    
    # 对发电机节点设置出力
    for i, gen in enumerate(env.net.gen.itertuples()):
        bus_idx = gen.bus
        p_g[bus_idx] = 50  # 设置发电机出力为50MW
    action = np.concatenate([v_g, p_g])
    # 执行一步
    obs, reward, done, info = env.step(action)
    print("\n执行一步后:")
    print("观测值:", obs)
    print("奖励值:", reward)
    print("是否结束:", done)
    
    # 渲染环境状态
    print("\n环境状态:")
    env.render()


