# evaluation_utils.py

import torch
from models import FSFLModel # 用于类型检查 (For type checking)
# CNNEncoder 将通过 model 参数间接传入 (CNNEncoder will be passed indirectly via the model parameter)

def evaluate_global_model(model, test_data, omniglot_train_dataset, support_indices_for_eval, global_label_mapping_for_eval):
    """
    在测试集上评估模型的准确率。
    Compute accuracy of a model on the test set.

    Args:
        model (torch.nn.Module): 要评估的模型 (可以是 FSFLModel 或 CNNEncoder)。
                                (The model to evaluate (can be FSFLModel or CNNEncoder).)
        test_data (list): 测试数据集 [(image_tensor, global_label), ...]。
                          (Test dataset.)
        omniglot_train_dataset (torchvision.datasets.Omniglot): 原始的 Omniglot 训练集，用于从支持索引中获取图像。
                                                               (Original Omniglot training set, to get images from support indices.)
        support_indices_for_eval (dict): 用于评估原型方法的支持集索引 {original_class_label: [indices...]}.
                                        (Support set indices for evaluating prototype method.)
        global_label_mapping_for_eval (dict): 原始类别标签到全局标签的映射 {original_class_label: global_label}.
                                             (Mapping from original class labels to global labels.)

    Returns:
        float: 模型在测试集上的准确率 (百分比)。
               (Accuracy of the model on the test set (percentage).)
    """
    model.eval() # 设置为评估模式 (Set to evaluation mode)
    correct = 0
    total = 0

    if not test_data:
        return 0.0 # 如果没有测试数据，准确率为0 (If no test data, accuracy is 0)

    if isinstance(model, FSFLModel):
        # FedAvg 模型 (带有分类器)
        # FedAvg model (has classifier)
        for img, label in test_data:
            img = img.unsqueeze(0)  # 添加批次维度 (add batch dimension)
            with torch.no_grad():
                output = model(img)
            pred = output.argmax(dim=1).item()
            if pred == label:
                correct += 1
            total += 1
    else:
        # 对于编码器 (基于原型的模型)，使用支持数据计算原型进行分类
        # For encoder (prototype-based model), use prototypes from support data for classification
        encoder = model # 此时 model 是一个 CNNEncoder
                       # At this point, model is a CNNEncoder
        
        # 使用支持数据为所有类别计算原型。
        # We need to use the support images from all clients (the training shots) to compute global prototypes.
        # 我们将利用之前准备的 support_indices 和 omniglot_train 数据集。
        # We'll leverage the support_indices and omniglot_train dataset prepared earlier.
        class_prototypes_eval = {} # 存储 {global_label: prototype_tensor}
                                   # Stores {global_label: prototype_tensor}
        
        encoder.eval() # 确保编码器在评估模式 (Ensure encoder is in eval mode)
        with torch.no_grad():
            for original_cls_label, idx_list in support_indices_for_eval.items():
                if original_cls_label not in global_label_mapping_for_eval:
                    # print(f"评估警告: 类别 {original_cls_label} 在支持索引中，但不在全局标签映射中。跳过。")
                    # print(f"Evaluation Warning: Class {original_cls_label} in support_indices but not in global_label_mapping. Skipping.")
                    continue  # 如果类别未使用，则跳过 (skip if class not used)
                
                global_label = global_label_mapping_for_eval[original_cls_label]
                
                imgs_for_proto = [omniglot_train_dataset[idx][0] for idx in idx_list]
                if not imgs_for_proto:
                    # print(f"评估警告: 类别 {original_cls_label} (全局 {global_label}) 没有支持图像用于原型计算。跳过。")
                    # print(f"Evaluation Warning: Class {original_cls_label} (global {global_label}) has no support images for prototype. Skipping.")
                    continue
                
                imgs_for_proto = torch.stack(imgs_for_proto)
                feats = encoder(imgs_for_proto)
                proto = feats.mean(dim=0)
                class_prototypes_eval[global_label] = proto
        
        if not class_prototypes_eval:
            print("评估错误: 未能为任何类别计算原型。准确率将为0。")
            print("Evaluation Error: Failed to compute prototypes for any class. Accuracy will be 0.")
            return 0.0

        # 现在通过最近原型对每个测试样本进行分类
        # Now classify each test sample by nearest prototype
        for img, label in test_data:
            img = img.unsqueeze(0) # 添加批次维度 (add batch dimension)
            with torch.no_grad():
                feat = encoder(img).squeeze(0) # 移除批次维度 (remove batch dimension)
            
            # 找到距离最小的类别
            # find class with minimum distance
            min_dist = float('inf')
            pred_class = -1 # 初始化为无效类别 (Initialize to an invalid class)
            
            if not class_prototypes_eval: # 双重检查，尽管上面有检查 (Double check, though checked above)
                total +=1
                continue

            for cls_global_label, proto in class_prototypes_eval.items():
                dist = torch.sum((feat - proto)**2).item()
                if dist < min_dist:
                    min_dist = dist
                    pred_class = cls_global_label
            
            if pred_class == label:
                correct += 1
            total += 1
            
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc