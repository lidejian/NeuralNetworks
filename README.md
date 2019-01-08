1. 需要安装的软件
	Python 3.6
	tensorflow 1.2

2. 目录介绍
	2.1 BLLIP：处理BLLIP数据相关代码
	2.3 data： 数据集存放目录
	2.5 model_trainer：训练模型代码 （这部分代码针对 implicit discourse recognition）
		2.5.1 single_task.py: 单任务神经网络模型定义
		2.5.2 train_single_task.py：训练单任务神经网络
		2.5.3 run_single_task.py： 调用train_single_task.py，使用指定的参数，训练模型
		2.5.4 tune_single_task.py： 对单任务模型进行调参
		2.5.5 multi_task.py: 多任务神经网络模型定义
		2.5.6 train_multi_task.py：训练多任务神经网络
		2.5.7 run_multi_task.py： train_multi_task.py，使用指定的参数，训练模型
		2.5.8 tune_multi_task.py： 对多任务模型进行调参
	2.6 model_trainer_cqa： 训练模型代码 （这部分代码针对 QQ 与 QA）, 结构与model_trainer一致
	2.7 record：各个模型的结构记录文件夹
	2.10 config.py：系统的配置文件
	2.11 dataset_processing.py：数据预处理代码
	2.12 record.py：结果记录代码
	2.13 sample.py：数据采样代码

3. 程序运行方式

	3.1 运行单任务模型：
		
		cd model_trainer && python run_single_task.py
		
		其中,可以设置run_single_task.py中变量来配置单任务的参数：
		a. model变量，来选择使用哪个模型进行训练，model可等于CNN、RNN、Attention_RNN1等。
		
		b. train_data_dir: 使用哪个目录下的数据来训练。例如：train_data_dir = config.DATA_PATH + "/four_way/PDTB_imp"，表示使用four_way/PDTB_imp下的数据来训练模型，即训练PDTB上的四分类模型。
		
	3.2 运行多任务模型
		
		cd model_trainer && python run_multi_task.py
		
		
		main_train_data_dir = config.DATA_PATH + "/four_way/PDTB_imp"
		aux_train_data_dir = config.DATA_PATH + "/four_way/BLLIP_exp"
		
		其中,可以设置run_multi_task.py中变量来配置多任务的参数：
		
		a. main_train_data_dir: 主任务的训练数据集路径
		b. aux_train_data_dir:  辅助任务的训练数据集路径
		例如：
			main_train_data_dir = config.DATA_PATH + "/four_way/PDTB_imp"
			aux_train_data_dir = config.DATA_PATH + "/four_way/BLLIP_exp"
			表示使用PDTB_imp的数据作为主任务数据集，使用BLLIP_exp（人工合成的隐性语篇关系数据集）作为辅助任务数据集。进行PDTB上的四分类实验（在four_way目录下，该目录下的数据是四分类的数据）

		
		c. model变量：主任务和辅助任务的共享方式。可以等于
			1) share_1: 等价共享
			2) share_2: 加权共享
			3）share_3: 门共享
		
		
		
	