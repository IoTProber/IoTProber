import os
import sys
import pandas as pd
import numpy as np
import logging

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from acquire_data import load_device_labels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)


def split_train_test(num_folds=5, random_seed=42):
    """
    对platform_data/csv目录中各个设备类型的数据进行训练集和测试集的划分。
    - 数据量 > 50000: 随机20000个样本作为测试集
    - 数据量 <= 50000: 按4:1比例划分(20%作为测试集)
    - 随机5次划分，测试集尽可能覆盖数据集的不同部分
    - 测试集保存至 evaluation/validation/test_{device_type}_{轮数}.csv
    - 训练集保存至 platform_data/local/{轮数}/ipraw_{device_type}.csv
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_dir = os.path.join(base_dir, "platform_data", "csv")
    validation_dir = os.path.join(base_dir, "evaluation", "validation")
    local_dir = os.path.join(csv_dir, "local")

    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)
    
    for fold in range(1, num_folds + 1):
        os.makedirs(os.path.join(local_dir, str(fold)), exist_ok=True)

    device_labels = load_device_labels()
    if not device_labels:
        logging.error("No device labels found!")
        return

    rng = np.random.RandomState(random_seed)

    for device_type in device_labels:
        csv_path = os.path.join(csv_dir, f"ipraw_{device_type}.csv")
        if not os.path.exists(csv_path):
            logging.warning(f"CSV file not found for {device_type}, skipping...")
            continue

        logging.info(f"Processing {device_type}...")
        df = pd.read_csv(csv_path)
        n = len(df)

        if n == 0:
            logging.warning(f"Empty dataset for {device_type}, skipping...")
            continue

        if n > 50000:
            test_size = 20000
        else:
            test_size = n // num_folds

        indices = np.arange(n)
        rng.shuffle(indices)

        for fold in range(num_folds):
            start = fold * test_size
            end = start + test_size

            if end <= n:
                test_indices = indices[start:end]
            else:
                test_indices = np.concatenate([indices[start:], indices[:end - n]])

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_indices] = False
            train_indices = np.where(train_mask)[0]

            test_df = df.iloc[test_indices]
            train_df = df.iloc[train_indices]

            test_path = os.path.join(validation_dir, f"test_{device_type}_{fold + 1}.csv")
            test_df.to_csv(test_path, index=False)

            train_path = os.path.join(local_dir, str(fold + 1), f"ipraw_{device_type}.csv")
            train_df.to_csv(train_path, index=False)

            logging.info(f"  {device_type} Fold {fold + 1}: train={len(train_df)}, test={len(test_df)}")

    logging.info("All device types split complete!")


if __name__ == "__main__":
    split_train_test()
