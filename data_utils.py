# data_utils.py

import random
from collections import defaultdict
import torch
from torchvision import datasets, transforms
from config import CLASSES_PER_CLIENT, SHOTS_PER_CLASS, IMAGE_SIZE, DATA_ROOT, RANDOM_SEED

# 设置随机种子以保证数据划分的可复现性
# Set random seed for reproducible data splits
random.seed(RANDOM_SEED)
# PyTorch的随机种子在 main.py 中设置

def load_omniglot_data():
    """
    加载 Omniglot 数据集的背景集 (训练集)。
    Loads the background set (training set) of the Omniglot dataset.

    Returns:
        torchvision.datasets.Omniglot: 加载的 Omniglot 数据集对象。
                                      (The loaded Omniglot dataset object.)
    """
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # 调整图像大小 (resize images)
        transforms.ToTensor()  # 转换为张量 (像素范围 [0,1]) (convert to tensor (pixel range [0,1]))
    ])
    # 使用 "background" 字符集作为类别的来源
    # We will use the "background" set of Omniglot characters as the pool of classes
    omniglot_train = datasets.Omniglot(root=DATA_ROOT, background=True, download=True, transform=transform)
    # torchvision中的Omniglot用目标类别索引 (从0开始) 标记图像。
    # 'background=True' 给我们提供了964个训练类别。
    # Omniglot in torchvision marks images with a target class index (0-based).
    # The 'background=True' gives us the set of 964 training classes.
    return omniglot_train

def prepare_federated_data(omniglot_train, num_clients):
    """
    准备联邦学习所需的数据，包括客户端数据、测试数据和类别映射。
    Prepares data for federated learning, including client data, test data, and class mappings.

    Args:
        omniglot_train (torchvision.datasets.Omniglot): 加载的 Omniglot 数据集。
                                                       (The loaded Omniglot dataset.)
        num_clients (int): 客户端数量。 (Number of clients.)

    Returns:
        tuple: 包含:
               - client_data (dict): 客户端数据字典 {client_id: [(image_tensor, global_label), ...]}
                                    (Client data dictionary.)
               - test_data (list): 测试数据集 [(image_tensor, global_label), ...]
                                   (Test dataset.)
               - support_indices (dict): 每个选定类别的支持集图像索引 {class_original_label: [indices...]}
                                         (Support set image indices for each selected class.)
               - global_label_mapping (dict): 原始类别标签到全局标签的映射 {class_original_label: global_label}
                                              (Mapping from original class labels to global labels.)
               - num_classes_total (int): 总的唯一类别数量。
                                          (Total number of unique classes.)
    """
    # 按类别分组图像索引，方便划分
    # Group image indices by class for easy splitting
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(omniglot_train):
        class_to_indices[label].append(idx)

    # 为了我们的模拟，限制类别子集
    # Limit to a subset of classes for our simulation
    all_classes = sorted(list(class_to_indices.keys())) # 排序以保证一致性 (Sort for consistency)
    num_classes_total = CLASSES_PER_CLIENT * num_clients
    if num_classes_total > len(all_classes):
        raise ValueError(
            f"请求的总类别数 ({num_classes_total}) 超过了 Omniglot 背景集中的可用类别数 ({len(all_classes)})。"
            f"请减少 NUM_CLIENTS 或 CLASSES_PER_CLIENT。"
            f"(Requested total classes ({num_classes_total}) exceeds available classes in Omniglot background set ({len(all_classes)})."
            f"Please reduce NUM_CLIENTS or CLASSES_PER_CLIENT.)"
        )
    selected_classes = all_classes[:num_classes_total]  # 为简单起见，选择前 N_total 个类别
                                                       # Pick the first N_total classes for simplicity

    # 将每个选定类别的索引划分为支持集和查询集
    # Split each selected class's indices into support and query sets
    support_indices = {}
    query_indices = {}
    for cls in selected_classes:
        indices = class_to_indices[cls]
        # 如果某个类别的图像数量少于需要量，则跳过 (Omniglot 类别应该各有20张图像)。
        # If the class has fewer images than needed, skip (Omniglot classes should have 20 each).
        if len(indices) < SHOTS_PER_CLASS + 1: # 至少需要 K 张用于支持集, 1 张用于查询集
                                               # Need at least K for support, 1 for query
            print(f"警告: 类别 {cls} 图像数量不足 ({len(indices)})，需要 {SHOTS_PER_CLASS + 1}。将跳过此类别。")
            print(f"Warning: Class {cls} has insufficient images ({len(indices)}), needs {SHOTS_PER_CLASS + 1}. Skipping this class.")
            selected_classes.remove(cls) # 从选定类别中移除，以免后续出错
                                         # Remove from selected classes to avoid errors later
            continue
        # 打乱索引并划分
        # Shuffle indices and split
        random.shuffle(indices)
        k = SHOTS_PER_CLASS
        support_indices[cls] = indices[:k]  # K 张支持图像 (K support images)
        query_indices[cls] = indices[k:]    # 剩余的用于查询/测试 (remaining for query/test)

    # 更新总类别数，以防有类别被跳过
    # Update total number of classes in case some were skipped
    num_classes_total = len(selected_classes)
    if num_classes_total < CLASSES_PER_CLIENT * num_clients:
        print(f"警告: 由于部分类别图像不足，实际使用的总类别数为 {num_classes_total}。")
        print(f"Warning: Due to insufficient images in some classes, the actual total number of classes used is {num_classes_total}.")
        # 根据实际使用的类别数调整客户端数量或每个客户端的类别数是一个复杂的问题，
        # 这里我们简单地继续，但最终的类别数可能少于预期。
        # Adjusting num_clients or classes_per_client based on actual classes is complex.
        # We'll proceed, but the final number of classes might be less than desired.
        # 为了简单起见，如果 selected_classes 不足以均匀分配，可能会导致错误。
        # 确保 num_clients * CLASSES_PER_CLIENT <= len(selected_classes)
        if num_clients * CLASSES_PER_CLIENT > num_classes_total:
             raise ValueError(
                f"调整后的总类别数 {num_classes_total} 不足以分配给 {num_clients} 个客户端，每个客户端 {CLASSES_PER_CLIENT} 个类别。"
                f"请检查原始数据或减少客户端/类别配置。"
                f"Adjusted total classes {num_classes_total} is not enough for {num_clients} clients with {CLASSES_PER_CLIENT} classes each."
                f"Please check raw data or reduce client/class configuration."
            )


    # 将类别分配给客户端
    # Assign classes to clients
    clients_classes = {}
    # 确保我们只使用实际可用的、未被跳过的类别
    # Ensure we only use actually available, non-skipped classes
    current_class_idx = 0
    for i in range(num_clients):
        cls_subset = []
        while len(cls_subset) < CLASSES_PER_CLIENT and current_class_idx < len(selected_classes):
            cls_candidate = selected_classes[current_class_idx]
            if cls_candidate in support_indices: # 确保这个类别没有因为样本不足而被跳过
                                                # Ensure this class wasn't skipped due to insufficient samples
                cls_subset.append(cls_candidate)
            current_class_idx += 1
        if len(cls_subset) < CLASSES_PER_CLIENT:
            print(f"警告: 客户端 {i} 分配到的类别数少于预期 ({len(cls_subset)} < {CLASSES_PER_CLIENT})。")
            print(f"Warning: Client {i} was assigned fewer classes than expected ({len(cls_subset)} < {CLASSES_PER_CLIENT}).")
        clients_classes[i] = cls_subset


    # 构建客户端数据集 (图像张量, 标签列表) 和测试集
    # Build client datasets (lists of (image_tensor, label)) and test set
    client_data = {i: [] for i in range(num_clients)}
    test_data = []
    global_label_mapping = {}  # 将原始类别标签映射到全局标签索引 (0...num_classes_total-1)
                               # Map class to global label index (0...num_classes_total-1)
    # 使用实际分配给客户端的类别来构建全局标签映射，以确保一致性
    # Use classes actually assigned to clients to build the global label mapping for consistency
    all_assigned_classes = sorted(list(set(cls for subset in clients_classes.values() for cls in subset)))

    for glabel, cls in enumerate(all_assigned_classes):
        global_label_mapping[cls] = glabel
    
    # 更新实际使用的总类别数
    # Update the actual total number of unique classes being used
    num_classes_total = len(all_assigned_classes)


    for client_id, class_list in clients_classes.items():
        for cls in class_list:
            if cls not in global_label_mapping: # 如果某个类别最终没有被任何客户端使用 (不太可能发生，但作为预防)
                                                # If a class wasn't used by any client (unlikely, but as a safeguard)
                continue
            # 为类别分配全局标签
            # assign global label for the class
            glabel = global_label_mapping[cls]
            # 将支持图像添加到客户端数据，并使用全局标签
            # add support images to client data with global label
            for idx in support_indices.get(cls, []):
                img, _ = omniglot_train[idx]
                client_data[client_id].append((img, glabel))
            # 将查询图像添加到测试集，并使用全局标签
            # add query images to test set with global label
            for idx in query_indices.get(cls, []):
                img, _ = omniglot_train[idx]
                test_data.append((img, glabel))

    # 健全性打印: 每个客户端的类别数和样本数，以及总测试样本数
    # Sanity print: number of classes and samples per client, and total test samples
    print("\n--- 数据准备摘要 (Data Preparation Summary) ---")
    for cid, data in client_data.items():
        if data: # 仅当客户端有数据时打印
                 # Print only if client has data
            labels = {lbl for (_, lbl) in data}
            print(f"客户端 (Client) {cid}: {len(labels)} 个类别 (classes), {len(data)} 个样本 (samples)")
        else:
            print(f"客户端 (Client) {cid}: 0 个类别 (classes), 0 个样本 (samples) (可能由于类别不足导致)")
            print(f"Client {cid}: 0 classes, 0 samples (possibly due to insufficient classes)")
    print(f"总测试样本数 (Total test samples): {len(test_data)}")
    print(f"使用的总唯一类别数 (Total unique classes used): {num_classes_total}")
    print("--- 数据准备完成 (Data Preparation Complete) ---\n")

    return client_data, test_data, support_indices, global_label_mapping, num_classes_total