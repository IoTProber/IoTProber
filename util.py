import binascii
import os
import sys
import json
import ast
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from subprocess import check_output, STDOUT

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def load_all_dev_labels():
    """
    Load all device labels that will be configured into the graph
    """
    device_label_list = []
    label_file_path = os.path.join(BASE_PATH, "rag_devices.json")
    if not os.path.exists(label_file_path):
        print(f"No device labels file found!")
        return None

    with open(label_file_path, "r") as f:
        dt = json.load(f)

    for device_type in dt.values():
        device_label_list.extend(device_type)

    return device_label_list


def load_new_dev_labels():
    device_label_list = []
    label_file_path = os.path.join(BASE_PATH, "new_devices.json")
    if not os.path.exists(label_file_path):
        print(f"No device labels file found!")
        return None

    with open(label_file_path, "r") as f:
        dt = json.load(f)

    for device_type in dt.values():
        device_label_list.extend(device_type)

    return device_label_list

def load_perspective_info():
    """加载 perspective 信息 """
    perspective_info_path = os.path.join(BASE_PATH, "perspective_info.json")
    if not os.path.exists(perspective_info_path):
        raise FileNotFoundError(f"perspective 信息文件不存在: {perspective_info_path}")
    
    with open(perspective_info_path, "r") as f:
        perspective_info = json.load(f)
    
    return perspective_info

def load_perspective_cluster_info():
    """加载 perspective 信息 """
    perspective_info_path = os.path.join(BASE_PATH, "perspective_name.json")
    if not os.path.exists(perspective_info_path):
        raise FileNotFoundError(f"perspective 信息文件不存在: {perspective_info_path}")
    
    with open(perspective_info_path, "r") as f:
        perspective_cluster_info = json.load(f)
    
    return perspective_cluster_info

def load_local_used_features():
    """加载本地使用的特征列表"""
    local_feature_path = os.path.join(BASE_PATH, "local_used_feature.txt")
    if not os.path.exists(local_feature_path):
        raise FileNotFoundError(f"本地特征文件不存在: {local_feature_path}")

    with open(local_feature_path, "r", encoding="utf-8") as f:
        local_used_features = [line.strip() for line in f.readlines() if line.strip()]

    return local_used_features

def preprocess_vector(vec, weights):
    """
    核心步骤：对向量进行加权，并进行 L2 归一化
    这样在使用 Inner Product (IP) 检索时，结果等同于加权余弦相似度
    """
    weighted_vec = vec * weights
    norm = np.linalg.norm(weighted_vec)
    return weighted_vec / norm if norm > 0 else weighted_vec

def convert_json_from_str(text):
    """将LLM输出转换成 json """
    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, text, re.DOTALL)
    
    clean_content = match.group(1) if match else text
    return json.loads(clean_content)

def hex_to_bit_list(hex_value):
    # 将十六进制数转换为二进制，并去掉前缀 '0b'
    binary_value = bin(hex_value)[2:].zfill(16)  # zfill(16) 确保输出为16位

    # 将二进制字符串转换为比特位列表
    bit_list = [int(bit) for bit in binary_value]

    return bit_list


def write_list_to_file(filepath, data_list):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(f"{item}\n")

def execute(command):
    """
    Executes a command on the local host.
    :param str command: the command to be executedi
    :return: returns the output of the STDOUT or STDERR
    """
    print("Shell command : {}".format(command))
    # command = "{}; exit 0".format(command)
    return check_output(command, stderr=STDOUT, shell=True).decode("utf-8")


def list_files_in_folder(directory: str):
    all_file_path = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_file_path.append(os.path.join(root, file))
    return all_file_path


def get_filename_without_extension(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def read_list_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def read_tuple_list_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [ast.literal_eval(line.strip()) for line in f.readlines()]


def check_extension(file_path: str, extension: str) -> bool:
    # 获取文件的后缀名
    _, ext = os.path.splitext(file_path)

    # 检查是否是 .csv 后缀
    if ext.lower() == extension:
        return True
    else:
        return False


def split_list_by_size(lst, size=2000):
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def decode_mixed_logs(raw: bytes) -> str:
    results = []

    try:
        results.append(raw.decode('utf-8'))
    except UnicodeDecodeError:
        pass  # 继续按字节扫描

    # 查找 UTF-8 解码失败点
    for i in range(len(raw)):
        try:
            prefix = raw[:i].decode('utf-8')
            suffix = raw[i:].decode('gbk')
            return prefix + suffix
        except UnicodeDecodeError:
            continue

    return repr(raw)


def process_labels(labels):
    processed = []
    for label in labels:
        # 去除末尾的 \r 和 \n
        clean_label = label.rstrip('\r\n')
        # 如果长度超过30，按每30字符加一个换行
        if len(clean_label) > 30:
            # 将字符串按每30字符分段并插入换行
            chunks = [clean_label[i:i+30] for i in range(0, len(clean_label), 30)]
            clean_label = '\n'.join(chunks)
        processed.append(clean_label)
    return processed


def parse_banner_hex(banner_hex: str):
    hex_str = banner_hex.strip().replace(" ", "").replace("\n", "")

    try:
        # banner_hex 转换成字符串
        raw_bytes = binascii.unhexlify(hex_str)
        http_text = raw_bytes.decode("utf-8", errors="ignore")
        return http_text

    except Exception as e:
        print(f"[ERROR] Transform hex banner to string fail!: {e}")
        return None

def chunk_text(text, max_length=500, overlap=50):
    """
    The chunk_text function is invoked in cluster.py as part of a resilience strategy during the generation of embeddings for IoT device traffic features. 
    When the embedding model fails to process a full text input (typically due to GPU memory limits), 
    the system attempts to break the text into smaller segments using chunk_text, embed each segment, and then average the resulting vectors.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

if __name__ == "__main__":
    pass