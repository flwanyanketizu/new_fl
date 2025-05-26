# main.py

import torch
import matplotlib.pyplot as plt
import random # 导入 random 以便设置种子 (Import random to set its seed)

# 从其他模块导入配置、函数和类
# Import configurations, functions, and classes from other modules
from config import (
    NUM_CLIENTS, CLASSES_PER_CLIENT, SHOTS_PER_CLASS, RANDOM_SEED,
    ROUNDS, LEARNING_RATE, FSFL_TEMPERATURE, FSFL_ALPHA
)
from data_utils import load_omniglot_data, prepare_federated_data
from models import CNNEncoder, FSFLModel
from federated_utils import (
    fedavg_local_train, fedfsl_local_train, fsfl_local_train, aggregate_average
)
from evaluation_utils import evaluate_global_model

def main():
    # --- 设置随机种子 (Set Random Seeds) ---
    # 注意: config.py 中已为 Python 内置 random 设置种子
    # Note: Python's built-in random seed is already set in config.py
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED) # 再次确保，以防其他地方未设置 (Ensure again, in case not set elsewhere)
    
    # --- 1. 加载和准备数据 (Load and Prepare Data) ---
    print("步骤 1: 加载 Omniglot 数据集...")
    print("Step 1: Loading Omniglot dataset...")
    omniglot_train_dataset = load_omniglot_data()
    
    print("\n步骤 2: 为联邦学习准备数据...")
    print("Step 2: Preparing data for Federated Learning...")
    client_data, test_data, support_indices, global_label_mapping, num_classes_total = \
        prepare_federated_data(omniglot_train_dataset, NUM_CLIENTS)

    if num_classes_total == 0:
        print("错误: 没有可用的类别进行训练。请检查数据准备步骤和配置。")
        print("Error: No classes available for training. Check data preparation steps and configuration.")
        return

    print(f"联邦学习中使用的总类别数 (Total classes used in FL): {num_classes_total}")

    # --- 2. 初始化全局模型 (Initialize Global Models) ---
    print("\n步骤 3: 初始化全局模型...")
    print("Step 3: Initializing global models...")

    # 对于 FedAvg，是完整的模型 (编码器+分类器)
    # For FedAvg, a full model (encoder+classifier)
    global_model_fedavg = FSFLModel(num_classes=num_classes_total)
    
    # 对于 FedFSL 和 FsFL，我们只使用编码器
    # For FedFSL and FsFL, we'll use just encoders
    global_encoder_fedfsl = CNNEncoder()
    # 从 FedAvg 模型的编码器部分复制初始权重，以保证公平比较
    # Copy initial weights from FedAvg model's encoder part for fair comparison
    global_encoder_fedfsl.load_state_dict(global_model_fedavg.encoder.state_dict()) 
    
    global_encoder_fsfl = CNNEncoder()
    global_encoder_fsfl.load_state_dict(global_model_fedavg.encoder.state_dict()) # 同样的初始权重 (Same initial weights)

    # --- 3. 追踪准确率历史 (Track Accuracy History) ---
    history_fedavg = []
    history_fedfsl = []
    history_fsfl = []

    # --- 4. 开始联邦训练 (Start Federated Training) ---
    print("\n步骤 4: 开始联邦训练...\n")
    print("Step 4: Starting Federated Training...\n")

    for r in range(1, ROUNDS + 1):
        print(f"--- 第 {r} 轮 (Round {r}) ---")
        
        # --- FedAvg 轮次 (FedAvg Round) ---
        print("FedAvg: 正在所有客户端上执行本地训练...")
        print("FedAvg: performing local training on all clients...")
        client_model_params_fedavg = []
        for cid in range(NUM_CLIENTS):
            if client_data[cid]: # 确保客户端有数据 (Ensure client has data)
                print(f"  [FedAvg] 客户端 (Client) {cid} 开始训练...")
                print(f"  [FedAvg] Client {cid} starting training...")
                local_params = fedavg_local_train(global_model_fedavg, client_data[cid], LEARNING_RATE, num_classes_total)
                client_model_params_fedavg.append(local_params)
                # fedavg_local_train 内部会打印损失 (fedavg_local_train prints loss internally)
            else:
                print(f"  [FedAvg] 客户端 (Client) {cid} 没有数据，跳过训练。")
                print(f"  [FedAvg] Client {cid} has no data, skipping training.")


        if client_model_params_fedavg: # 只有在有客户端更新时才聚合 (Aggregate only if there were client updates)
            new_global_state_fedavg = aggregate_average(client_model_params_fedavg)
            global_model_fedavg.load_state_dict(new_global_state_fedavg)
        
        acc_fedavg = evaluate_global_model(global_model_fedavg, test_data, omniglot_train_dataset, support_indices, global_label_mapping)
        history_fedavg.append(acc_fedavg)
        print(f" FedAvg 第 {r} 轮后全局准确率 (Global Accuracy after round {r}): {acc_fedavg:.2f}%\n")

        # --- FedFSL 轮次 (FedFSL Round) ---
        print("FedFSL: 正在所有客户端上执行本地训练...")
        print("FedFSL: performing local training on all clients...")
        client_encoder_params_fedfsl = []
        for cid in range(NUM_CLIENTS):
            if client_data[cid]:
                print(f"  [FedFSL] 客户端 (Client) {cid} 开始训练...")
                print(f"  [FedFSL] Client {cid} starting training...")
                local_params = fedfsl_local_train(global_encoder_fedfsl, client_data[cid], LEARNING_RATE)
                client_encoder_params_fedfsl.append(local_params)
                # fedfsl_local_train 内部会打印损失 (fedfsl_local_train prints loss internally)
            else:
                print(f"  [FedFSL] 客户端 (Client) {cid} 没有数据，跳过训练。")
                print(f"  [FedFSL] Client {cid} has no data, skipping training.")

        if client_encoder_params_fedfsl:
            new_enc_state_fedfsl = aggregate_average(client_encoder_params_fedfsl)
            global_encoder_fedfsl.load_state_dict(new_enc_state_fedfsl)

        acc_fedfsl = evaluate_global_model(global_encoder_fedfsl, test_data, omniglot_train_dataset, support_indices, global_label_mapping)
        history_fedfsl.append(acc_fedfsl)
        print(f" FedFSL 第 {r} 轮后全局准确率 (Global Accuracy after round {r}): {acc_fedfsl:.2f}%\n")

        # --- FsFL 轮次 (FsFL Round) ---
        print("FsFL: 正在所有客户端上执行本地训练...")
        print("FsFL: performing local training on all clients...")
        client_encoder_params_fsfl = []
        total_proto_loss_fsfl = 0
        total_kd_loss_fsfl = 0
        num_fsfl_clients_trained = 0

        for cid in range(NUM_CLIENTS):
            if client_data[cid]:
                print(f"  [FsFL] 客户端 (Client) {cid} 开始训练...")
                print(f"  [FsFL] Client {cid} starting training...")
                local_params, proto_loss, kd_loss = fsfl_local_train(
                    global_encoder_fsfl, client_data[cid], 
                    FSFL_TEMPERATURE, FSFL_ALPHA, LEARNING_RATE
                )
                client_encoder_params_fsfl.append(local_params)
                total_proto_loss_fsfl += proto_loss
                total_kd_loss_fsfl += kd_loss
                num_fsfl_clients_trained +=1
                print(f"  [FsFL] 客户端 (Client) {cid} 原型损失 (proto_loss) = {proto_loss:.4f}, 知识蒸馏损失 (kd_loss) = {kd_loss:.4f}")
            else:
                print(f"  [FsFL] 客户端 (Client) {cid} 没有数据，跳过训练。")
                print(f"  [FsFL] Client {cid} has no data, skipping training.")
        
        if num_fsfl_clients_trained > 0:
            avg_proto_loss = total_proto_loss_fsfl / num_fsfl_clients_trained
            avg_kd_loss = total_kd_loss_fsfl / num_fsfl_clients_trained
            print(f"  [FsFL] 平均原型损失 (Avg proto_loss) = {avg_proto_loss:.4f}, 平均知识蒸馏损失 (Avg kd_loss) = {avg_kd_loss:.4f}")


        if client_encoder_params_fsfl:
            new_enc_state_fsfl = aggregate_average(client_encoder_params_fsfl)
            global_encoder_fsfl.load_state_dict(new_enc_state_fsfl)
        
        acc_fsfl = evaluate_global_model(global_encoder_fsfl, test_data, omniglot_train_dataset, support_indices, global_label_mapping)
        history_fsfl.append(acc_fsfl)
        print(f" FsFL 第 {r} 轮后全局准确率 (Global Accuracy after round {r}): {acc_fsfl:.2f}%\n")

    # --- 5. 绘制结果 (Plot Results) ---
    print("\n步骤 5: 绘制准确率曲线...")
    print("Step 5: Plotting accuracy curves...")
    plt.figure(figsize=(10, 6)) # 调整图形大小以便更好地显示 (Adjust figure size for better display)
    plt.plot(range(1, ROUNDS + 1), history_fedavg, marker='o', linestyle='-', label='FedAvg')
    plt.plot(range(1, ROUNDS + 1), history_fedfsl, marker='s', linestyle='--', label='FedFSL (Prototype-based)')
    plt.plot(range(1, ROUNDS + 1), history_fsfl, marker='^', linestyle='-.', label='FsFL (Prototype + KD)')
    
    plt.title('全局模型准确率随轮数变化 (Global Model Accuracy per Round)')
    plt.xlabel('联邦学习轮数 (Federated Learning Round)')
    plt.ylabel('测试准确率 (%) (Test Accuracy (%))')
    plt.xticks(range(1, ROUNDS + 1)) #确保x轴刻度为整数轮数 (Ensure x-axis ticks are integer rounds)
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # 调整布局以防止标签重叠 (Adjust layout to prevent label overlap)
    
    # 保存图形到文件 (Save the plot to a file)
    plot_filename = "federated_learning_accuracy_comparison.png"
    plt.savefig(plot_filename)
    print(f"准确率曲线图已保存为 (Accuracy plot saved as): {plot_filename}")
    
    # 显示图形 (Show the plot)
    # plt.show() # 在某些环境中可能需要注释掉，例如在无头服务器上运行时
               # May need to be commented out in some environments, e.g., when running on a headless server

    print("\n--- 联邦学习模拟完成 (Federated Learning Simulation Complete) ---")

if __name__ == '__main__':
    main()