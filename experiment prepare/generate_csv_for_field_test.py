import pandas as pd
import os

def process_csv_files(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # 检查'v5_mpc_base2'列是否存在
            if 'v5_mpc_base2' in df.columns:
                # 获取'v5_mpc_base2'列，转换为列表
                #data_list = df['v5_mpc_base2'].dropna().tolist()
                data_list = df['v5_mpc_base2'].iloc[:230].dropna().tolist()

                # 只有当列表中有数据时才进行处理
                if data_list:
                    # 获取第一行的值，重复20次添加到列表前
                    first_value = data_list[0]
                    # 列表前添加
                    prefixed_list = [first_value] * 20 + data_list

                    # 获取最后一行的值，重复10次添加到列表后
                    last_value = data_list[-1]
                    print('last_value', last_value)
                    # 列表后添加
                    final_list = prefixed_list + [last_value] * 10
                else:
                    # 如果data_list为空，则创建一个空列表
                    final_list = []

                # 使用原始文件名中的第一个数字来命名新文件
                first_number = filename.split('_')[0]
                new_filename = f"{first_number}.csv"
                output_path = os.path.join(output_folder, new_filename)

                # 保存修改后的列表为CSV文件
                pd.DataFrame(final_list).to_csv(output_path, index=False, header=False)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"'v5_mpc_base2' column not found in {filename}.")
        else:
            pass

# 设置输入和输出文件夹路径
input_folder = './data/NGSIM_I80_results/'
output_folder = './data/NGSIM_I80_field_test/'

# 调用函数处理文件
process_csv_files(input_folder, output_folder)
