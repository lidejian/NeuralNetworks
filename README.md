1. ��Ҫ��װ�����
	Python 3.6
	tensorflow 1.2

2. Ŀ¼����
	2.1 BLLIP������BLLIP������ش���
	2.3 data�� ���ݼ����Ŀ¼
	2.5 model_trainer��ѵ��ģ�ʹ��� ���ⲿ�ִ������ implicit discourse recognition��
		2.5.1 single_task.py: ������������ģ�Ͷ���
		2.5.2 train_single_task.py��ѵ��������������
		2.5.3 run_single_task.py�� ����train_single_task.py��ʹ��ָ���Ĳ�����ѵ��ģ��
		2.5.4 tune_single_task.py�� �Ե�����ģ�ͽ��е���
		2.5.5 multi_task.py: ������������ģ�Ͷ���
		2.5.6 train_multi_task.py��ѵ��������������
		2.5.7 run_multi_task.py�� train_multi_task.py��ʹ��ָ���Ĳ�����ѵ��ģ��
		2.5.8 tune_multi_task.py�� �Զ�����ģ�ͽ��е���
	2.6 model_trainer_cqa�� ѵ��ģ�ʹ��� ���ⲿ�ִ������ QQ �� QA��, �ṹ��model_trainerһ��
	2.7 record������ģ�͵Ľṹ��¼�ļ���
	2.10 config.py��ϵͳ�������ļ�
	2.11 dataset_processing.py������Ԥ�������
	2.12 record.py�������¼����
	2.13 sample.py�����ݲ�������

3. �������з�ʽ

	3.1 ���е�����ģ�ͣ�
		
		cd model_trainer && python run_single_task.py
		
		����,��������run_single_task.py�б��������õ�����Ĳ�����
		a. model��������ѡ��ʹ���ĸ�ģ�ͽ���ѵ����model�ɵ���CNN��RNN��Attention_RNN1�ȡ�
		
		b. train_data_dir: ʹ���ĸ�Ŀ¼�µ�������ѵ�������磺train_data_dir = config.DATA_PATH + "/four_way/PDTB_imp"����ʾʹ��four_way/PDTB_imp�µ�������ѵ��ģ�ͣ���ѵ��PDTB�ϵ��ķ���ģ�͡�
		
	3.2 ���ж�����ģ��
		
		cd model_trainer && python run_multi_task.py
		
		
		main_train_data_dir = config.DATA_PATH + "/four_way/PDTB_imp"
		aux_train_data_dir = config.DATA_PATH + "/four_way/BLLIP_exp"
		
		����,��������run_multi_task.py�б��������ö�����Ĳ�����
		
		a. main_train_data_dir: �������ѵ�����ݼ�·��
		b. aux_train_data_dir:  ���������ѵ�����ݼ�·��
		���磺
			main_train_data_dir = config.DATA_PATH + "/four_way/PDTB_imp"
			aux_train_data_dir = config.DATA_PATH + "/four_way/BLLIP_exp"
			��ʾʹ��PDTB_imp��������Ϊ���������ݼ���ʹ��BLLIP_exp���˹��ϳɵ�������ƪ��ϵ���ݼ�����Ϊ�����������ݼ�������PDTB�ϵ��ķ���ʵ�飨��four_wayĿ¼�£���Ŀ¼�µ��������ķ�������ݣ�

		
		c. model������������͸�������Ĺ���ʽ�����Ե���
			1) share_1: �ȼ۹���
			2) share_2: ��Ȩ����
			3��share_3: �Ź���
		
		
		
	