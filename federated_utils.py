# federated_utils.py

import torch
import torch.nn.functional as F
from models import CNNEncoder, FSFLModel # 从 models.py 导入模型定义 (Import model definitions from models.py)
from collections import OrderedDict

def fedavg_local_train(global_model, client_local_data, lr, num_classes_total):
    """
    在每个客户端上执行 FedAvg 本地训练，并返回客户端模型的状态字典。
    Perform FedAvg local training on each client and return the client models' state dicts.

    Args:
        global_model (FSFLModel): 当前的全局模型。 (Current global model.)
        client_local_data (list): 当前客户端的数据 [(image_tensor, global_label), ...]。
                                  (Current client's data.)
        lr (float): 本地训练的学习率。 (Learning rate for local training.)
        num_classes_total (int): 数据集中的总类别数，用于初始化本地模型。
                                 (Total number of classes in the dataset, for initializing local model.)


    Returns:
        OrderedDict: 更新后的本地模型参数 (state_dict)。
                     (Updated local model parameters (state_dict).)
    """
    global_model.train() # 设置为训练模式 (Set to training mode)
    
    # 将本地模型初始化为全局模型的副本
    # Initialize local model as a copy of global model
    local_model = FSFLModel(num_classes=num_classes_total)
    local_model.load_state_dict(global_model.state_dict())
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    if not client_local_data: # 如果客户端没有数据，则直接返回原始全局模型参数
                              # If client has no data, return original global model parameters
        return global_model.state_dict()

    # 准备数据张量
    # Prepare data tensors
    images = torch.stack([img for (img, _) in client_local_data])
    labels = torch.tensor([lbl for (_, lbl) in client_local_data])

    # 在本地数据上进行单轮 (或单批次) 训练
    # Single epoch (or single batch) training on local data
    optimizer.zero_grad()
    outputs = local_model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # 返回更新后的模型参数
    # Save updated model parameters
    return local_model.state_dict()


def fedfsl_local_train(global_encoder, client_local_data, lr):
    """
    在每个客户端上执行 FedFSL 本地训练 (基于原型)，并返回更新后的编码器参数。
    Perform FedFSL local training (prototype-based) on each client and return updated encoder params.

    Args:
        global_encoder (CNNEncoder): 当前的全局编码器。 (Current global encoder.)
        client_local_data (list): 当前客户端的数据 [(image_tensor, global_label), ...]。
                                  (Current client's data.)
        lr (float): 本地训练的学习率。 (Learning rate for local training.)

    Returns:
        OrderedDict: 更新后的本地编码器参数 (state_dict)。
                     (Updated local encoder parameters (state_dict).)
    """
    global_encoder.train() # 设置为训练模式 (Set to training mode)

    # 将全局编码器权重复制到本地编码器
    # Copy global encoder weights to local encoder
    local_encoder = CNNEncoder()
    local_encoder.load_state_dict(global_encoder.state_dict())
    optimizer = torch.optim.SGD(local_encoder.parameters(), lr=lr)

    if not client_local_data: # 如果客户端没有数据
                              # If client has no data
        return global_encoder.state_dict()

    # 准备数据张量
    # Prepare data tensors
    images = torch.stack([img for (img, _) in client_local_data])
    labels_list = [lbl for (_, lbl) in client_local_data]
    labels = torch.tensor(labels_list)

    # 计算支持集的类别原型
    # Compute class prototypes for support set
    # (这里我们将所有本地数据都视为支持集，因为 K 很小。)
    # (Here we treat all local data as support since K is small.)
    # (如果 K < 本地数据总数，可以将其划分为训练用的支持集/查询集。)
    # (If K < total local, could split into support/query sets for training.)
    features = local_encoder(images)  # [样本数, 特征维度] ([num_samples, feat_dim])
    
    # 为此客户端数据中存在的每个类别计算原型
    # Compute prototype for each class present in this client's data
    class_prototypes = {}
    unique_labels = labels.unique()
    for lbl_val in unique_labels:
        lbl_item = lbl_val.item() # 将Tensor标量转换为Python数字 (Convert Tensor scalar to Python number)
        class_mask = (labels == lbl_val)
        # 原型 = 类别 'lbl' 特征的均值
        # prototype = mean of features for class 'lbl'
        proto = features[class_mask].mean(dim=0)
        class_prototypes[lbl_item] = proto
    
    # 现在计算原型损失: 每个样本到每个原型的距离
    # Now compute prototypical loss: distance of each sample to each prototype
    # 我们将使用负的L2距离平方作为softmax分类的logits
    # We'll use negative squared L2 distance as logits for softmax classification
    dists = []
    prototype_classes = sorted(list(class_prototypes.keys())) # 确保原型顺序一致 (Ensure consistent prototype order)

    for feat in features:
        # 该样本到每个类别原型的距离
        # distances from this sample to each class prototype
        sample_dists = []
        for cls_key in prototype_classes: # 按照排序后的类别顺序计算距离 (Calculate distances in sorted class order)
            proto = class_prototypes[cls_key]
            # L2距离平方
            # squared L2 distance
            dist = torch.sum((feat - proto) ** 2)
            # 我们使用负距离作为相似度得分 (越高 = 越近)
            # We use negative distance as a similarity score (higher = closer)
            sample_dists.append(-dist)
        dists.append(torch.stack(sample_dists))
    
    if not dists: # 如果没有计算出距离 (例如，如果没有原型)
                  # If no distances were computed (e.g., no prototypes)
        print("  [FedFSL] 警告: 客户端没有有效的原型来计算损失。跳过更新。")
        print("  [FedFSL] Warning: Client has no valid prototypes to compute loss. Skipping update.")
        return global_encoder.state_dict()
        
    dists = torch.stack(dists)  # 形状 [样本数, 本地类别数] (shape [num_samples, num_classes_local])
    
    # 创建目标索引 (每个样本的真实类别在本地原型列表排序中的索引)
    # Create target indices (each sample's true class index in the local prototype list ordering)
    # 将全局标签映射到原型列表中的索引
    # Map global label to index in prototype list
    target_indices = []
    for lbl_val in labels:
        lbl_item = lbl_val.item()
        try:
            target_idx = prototype_classes.index(lbl_item) # 使用排序后的列表获取索引 (Use sorted list to get index)
            target_indices.append(target_idx)
        except ValueError:
            # 如果某个标签不在原型类别中 (理论上不应该发生，因为原型是从这些标签生成的)
            # If a label is not in prototype_classes (should not happen as prototypes are generated from these labels)
            print(f"  [FedFSL] 错误: 标签 {lbl_item} 未在原型类别 {prototype_classes} 中找到。")
            print(f"  [FedFSL] Error: Label {lbl_item} not found in prototype classes {prototype_classes}.")
            # 这种情况下，可以选择跳过或引发错误
            # In this case, one might choose to skip or raise an error
            return global_encoder.state_dict() # 跳过此客户端的更新 (Skip update for this client)


    target_indices = torch.tensor(target_indices, device=dists.device)

    # 在距离上计算交叉熵损失 (将它们视为logits)
    # Compute cross-entropy loss on the distances (treating them as logits)
    loss = F.cross_entropy(dists, target_indices)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 返回更新后的编码器参数
    # Save updated encoder parameters
    return local_encoder.state_dict()


def fsfl_local_train(global_encoder, client_local_data, T, alpha, lr):
    """
    在每个客户端上执行 FsFL 本地训练 (基于原型 + 知识蒸馏)。
    Perform FsFL local training on each client (prototype-based + knowledge distillation).

    Args:
        global_encoder (CNNEncoder): 当前的全局编码器 (作为教师模型)。
                                     (Current global encoder (acts as teacher model).)
        client_local_data (list): 当前客户端的数据 [(image_tensor, global_label), ...]。
                                  (Current client's data.)
        T (float): 知识蒸馏的温度参数。 (Temperature for distillation.)
        alpha (float): 蒸馏损失与原型损失的权重。 (Weight for distillation loss vs prototype loss.)
        lr (float): 本地训练的学习率。 (Learning rate for local training.)

    Returns:
        OrderedDict: 更新后的本地编码器 (学生模型) 参数 (state_dict)。
                     (Updated local encoder (student model) parameters (state_dict).)
    """
    global_encoder.train() # 全局编码器 (用于初始化学生) 应处于训练模式以允许参数更新
                           # Global encoder (for initializing student) should be in train mode to allow param updates
    
    # 我们将传入的 global_encoder 视为教师模型 (为其创建一个副本用于知识蒸馏)
    # We will treat the passed global_encoder as the teacher (copy it for KD usage)
    teacher_encoder = CNNEncoder()
    teacher_encoder.load_state_dict(global_encoder.state_dict())
    teacher_encoder.eval()  # 教师模型在本地更新期间保持固定 (teacher remains fixed during local update)

    # 从全局权重初始化学生 (本地) 编码器
    # Initialize student (local) encoder from global weights
    student_encoder = CNNEncoder()
    student_encoder.load_state_dict(global_encoder.state_dict()) # 学生从当前全局状态开始 (Student starts from current global state)
    student_encoder.train() # 学生模型需要训练 (Student model needs to be trained)
    optimizer = torch.optim.SGD(student_encoder.parameters(), lr=lr)

    if not client_local_data: # 如果客户端没有数据
                              # If client has no data
        return global_encoder.state_dict() # 返回原始（或学生初始化时的）编码器参数
                                          # Return original (or student's initial) encoder parameters

    # 准备数据
    # Prepare data
    images = torch.stack([img for (img, _) in client_local_data])
    labels_list = [lbl for (_, lbl) in client_local_data]
    labels = torch.tensor(labels_list)

    # --- 教师模型处理 (Teacher Model Processing) ---
    # 计算教师模型在本地数据上的输出概率 (用于知识蒸馏)。
    # Teacher outputs feature embeddings; to get class probabilities, compute distances to prototypes using teacher features.
    # 首先获取教师特征和原型 (就像我们下面为学生模型做的那样)。
    # First get teacher features and prototypes (like we do for student below).
    with torch.no_grad(): # 教师模型不进行梯度更新 (No gradient updates for teacher model)
        teacher_feats = teacher_encoder(images)
    
    # 在教师特征空间中计算原型
    # Compute prototypes in teacher feature space
    teacher_prototypes = {}
    unique_labels = labels.unique()
    for lbl_val in unique_labels:
        lbl_item = lbl_val.item()
        mask = (labels == lbl_val)
        proto = teacher_feats[mask].mean(dim=0)
        teacher_prototypes[lbl_item] = proto
    
    # 计算每个样本到每个原型的教师距离 logits (仅针对本地类别)
    # Compute teacher distance-based logits for each sample to each prototype (over local classes only)
    teacher_logits_list = []
    # 确保原型顺序一致，这对于后续的 softmax 和 KL散度至关重要
    # Ensure consistent prototype order, crucial for subsequent softmax and KL divergence
    proto_classes_teacher = sorted(list(teacher_prototypes.keys())) 

    if not proto_classes_teacher: # 如果教师没有原型 (例如，客户端数据为空或只有一个样本无法形成有意义的原型)
                                  # If teacher has no prototypes (e.g. client data was empty or only one sample)
        print("  [FsFL] 警告: 教师模型没有有效的原型。跳过此客户端的蒸馏。")
        print("  [FsFL] Warning: Teacher model has no valid prototypes. Skipping distillation for this client.")
        # 仅使用原型损失进行更新 (如果可能) 或直接返回
        # Update with only prototype loss (if possible) or return directly
        # 为简单起见，这里我们只执行原型损失部分 (如果学生能计算)
        # For simplicity, here we'd only perform prototype loss part if student can compute
        # 但由于教师logits为空，我们不能计算kd_loss，所以alpha将不适用
        # However, since teacher_logits are empty, we can't compute kd_loss, so alpha would not apply well.
        # 一个更简单的处理方式是，如果教师无法产出，则此客户端不参与FsFL更新。
        # A simpler handling: if teacher cannot produce output, this client does not participate in FsFL update.
        return global_encoder.state_dict()


    for feat in teacher_feats:
        logit_row = []
        for cls_key in proto_classes_teacher:
            proto = teacher_prototypes[cls_key]
            dist = torch.sum((feat - proto) ** 2)
            logit_row.append(-dist)
        teacher_logits_list.append(torch.stack(logit_row))
    
    teacher_logits = torch.stack(teacher_logits_list)  # [样本数, 本地类别数] ([num_samples, num_local_classes])
    # 教师的软概率 (使用温度 T)
    # Soft probabilities from teacher (using temperature T)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)

    # --- 学生模型更新 (Student Model Update) ---
    optimizer.zero_grad()
    # 学生前向传播 (获取特征和原型)
    # Student forward (get features and prototypes)
    student_feats = student_encoder(images)
    student_prototypes = {}
    # 学生模型的原型类别应与教师模型的原型类别顺序一致
    # Student model's prototype classes should align with teacher's for consistency in loss calculation
    proto_classes_student = proto_classes_teacher # 使用相同的类别和顺序 (Use the same classes and order)

    for lbl_val in unique_labels: # 仍然基于原始标签来创建原型 (Still create prototypes based on original labels)
        lbl_item = lbl_val.item()
        if lbl_item not in proto_classes_student: # 理论上不应该发生，因为 unique_labels 来自于 labels
                                                 # Theoretically shouldn't happen as unique_labels come from labels
            continue
        mask = (labels == lbl_val)
        proto = student_feats[mask].mean(dim=0)
        student_prototypes[lbl_item] = proto
    
    # 计算学生 logits (距离) 到本地原型
    # Compute student logits (distances) to local prototypes
    student_logits_list = []
    for feat in student_feats:
        logit_row = []
        for cls_key in proto_classes_student: # 使用与教师相同的类别顺序 (Use same class order as teacher)
            proto = student_prototypes[cls_key]
            dist = torch.sum((feat - proto) ** 2)
            logit_row.append(-dist)
        student_logits_list.append(torch.stack(logit_row))
    student_logits = torch.stack(student_logits_list)
    
    # 计算原型分类损失 (与真实标签的交叉熵)
    # Compute prototype classification loss (cross-entropy with true labels)
    # 目标索引需要映射到 proto_classes_student 的索引
    # Target indices need to map to indices of proto_classes_student
    target_indices_list = []
    for lbl_val in labels:
        lbl_item = lbl_val.item()
        try:
            target_idx = proto_classes_student.index(lbl_item)
            target_indices_list.append(target_idx)
        except ValueError:
             # 应该不会发生，因为 proto_classes_student 是从 labels 的 unique 值构建的
             # Should not happen as proto_classes_student is built from unique values of labels
            print(f"  [FsFL] 学生损失错误: 标签 {lbl_item} 未在学生原型类别 {proto_classes_student} 中找到。")
            print(f"  [FsFL] Student loss error: Label {lbl_item} not found in student prototype classes {proto_classes_student}.")
            return student_encoder.state_dict() # 或 global_encoder.state_dict()
            
    target_indices = torch.tensor(target_indices_list, device=student_logits.device)
    proto_loss = F.cross_entropy(student_logits, target_indices)
    
    # 计算学生和教师概率之间的知识蒸馏损失 (针对本地类别)
    # Compute knowledge distillation loss between student & teacher probabilities (for local classes)
    student_log_probs = F.log_softmax(student_logits / T, dim=1) # 学生使用 log_softmax
                                                                # Student uses log_softmax

    # KL散度从 teacher_probs 到 student_probs (注意: KD损失通常使用教师作为 "目标")
    # KL divergence from teacher_probs to student_probs (note: KD loss often uses teacher as "target")
    # PyTorch的 F.kl_div(input, target) 期望 input 是 log-probabilities, target 是 probabilities.
    # reduction='batchmean' 将按批次大小和类别数进行平均 (如果适用)。
    # 'batchmean' averages over batch and (if applicable) class dimension.
    # 对于 (N,C) 和 (N,C) 的输入，它计算 sum(target_i * (log(target_i) - input_i)) / N
    # For inputs (N,C) and (N,C), it computes sum(target_i * (log(target_i) - input_i)) / N
    # 我们想要的是 D_KL(P_teacher || P_student) = sum(P_teacher * log(P_teacher / P_student))
    # = sum(P_teacher * (logP_teacher - logP_student))
    # F.kl_div(student_log_probs, teacher_probs) 计算的是 sum(teacher_probs * (log_teacher_probs - student_log_probs))
    # 这与标准的KD损失形式 sum(teacher_probs * log(teacher_probs)) - sum(teacher_probs * student_log_probs) 不同
    # 通常的KD损失是 -sum(teacher_probs * student_log_probs) 或者 CE(student_logits/T, teacher_logits/T) * T^2
    # 这里我们使用 F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')，这是一种常见的实现方式
    # (KL divergence: student_probs.log() vs teacher_probs)
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T) # 乘以 T^2 与 CE(logits/T) 等价
                                                                                       # Multiply by T^2 to make it equivalent to CE(logits/T)
    
    # 组合损失
    # Combine losses
    loss = proto_loss * (1 - alpha) + kd_loss * alpha
    loss.backward()
    optimizer.step()

    return student_encoder.state_dict(), proto_loss.item(), kd_loss.item()


def aggregate_average(param_list):
    """
    平均一个状态字典 (模型参数) 列表，并返回平均后的状态字典。
    Average a list of state_dicts (model parameters) and return the averaged state_dict.

    Args:
        param_list (list): 包含模型 state_dict 的列表。 (List containing model state_dicts.)

    Returns:
        OrderedDict: 平均后的模型参数 (state_dict)。
                     (Averaged model parameters (state_dict).)
    """
    avg_state = OrderedDict()
    if not param_list:
        return avg_state
    
    # 用第一个模型的参数初始化 avg_state (深拷贝)
    # initialize avg_state with the first model's params (deep copy)
    for key, val in param_list[0].items():
        avg_state[key] = val.clone().float() # 确保为浮点数以进行平均 (Ensure float for averaging)
    
    # 累加所有其他模型的参数
    # add up all other model parameters
    for state in param_list[1:]:
        for key, val in state.items():
            if key in avg_state:
                avg_state[key] += val.float() # 确保为浮点数 (Ensure float)
            else: # 理论上不应该发生，如果所有模型结构相同
                  # Theoretically should not happen if all models have the same structure
                avg_state[key] = val.clone().float()
    
    # 除以模型数量
    # divide by number of models
    num = len(param_list)
    if num > 0:
        for key in avg_state:
            avg_state[key] /= num
            
    return avg_state