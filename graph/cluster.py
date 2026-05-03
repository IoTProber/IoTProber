import os
import re
import sys
import time
import shutil
import logging
import argparse

import json
import numpy
import hdbscan
# from cuml.cluster import HDBSCAN   # 使用NVIDIA RAPIDS 加速
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from geopy.geocoders import Nominatim
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from google import genai
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

import torch, gc
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm import LLM
from util import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.8"

class GraphClustering:
    def __init__(self, gpu=1):
        self.gpu = gpu
        self.base_path = os.path.dirname(os.path.dirname(__file__))
        # self.data_path = os.path.join(self.base_path, "rag_data")
        self.data_path = os.path.join(self.base_path, "platform_data")
        # self.save_path = os.path.join(self.data_path, "csv")
        self.save_path = os.path.join(self.data_path, "csv/local/1")
        self.embedding_model_path = os.path.join(self.base_path, "qwen3_embedding_06b")
        self.geolocator = Nominatim(user_agent="abcd")
        self.platform_feature_cols = ["ip", "as-asn", "as-name", "as-bgp_prefix", "as-country_code", "as-info", 
                                    "loc-latitude", "loc-longitude", "loc-continent", "loc-country", "loc-country_code", "loc-province", "loc-city", "loc-postal_code", "loc-timezone", "loc-info", 
                                    "dns-reverse", 
                                    "whois-network-handle", "whois-network-name", "whois-organization-handle", "whois-organization-name", "whois-info",
                                    "os-vendor", "os-product", "os-version", "os-info", 
                                    "service-distribution", 
                                    "sw-vendors", "sw-products", "sw-versions", "sw-info", 
                                    "hw-vendors", "hw-products", "hw-versions", "hw-info",  
                                    "cert-fingerprints", "cert-subjects", "cert-issuers", "cert-info", "tls-versions", 
                                    "http-bodys", "http-tags", "http-favicon-urls", "http-favicon-hashes", "http-part-info", "http-info"]
       
        self.entity_graph_path = os.path.join(self.base_path, "entity_graph")
        
        self.perspective_info_config = load_perspective_info()
        self.perspective_cluster_info = load_perspective_cluster_info()
        self.perspective_names = list(self.perspective_cluster_info.keys())

        self.except_perspective = ["hpart", "http"]
        self.local_retrieval_perspective_names = [
            p for p in self.perspective_names if p not in self.except_perspective
        ]

        self.device_labels = load_all_dev_labels()
        self.llm = LLM()
        self.initialize()
        
    def initialize(self):
        if self.gpu != -1:
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_path,
                                                        # model_kwargs={"device": "cpu"},
                                                        model_kwargs={"device": self.gpu},
                                                        encode_kwargs={"normalize_embeddings": True,
                                                                    "batch_size": 4},
                                                        query_encode_kwargs={"normalize_embeddings": True,
                                                                            "batch_size": 1})
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        else:
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_path,
                                                        model_kwargs={"device": "cpu"},
                                                        # model_kwargs={"device": self.gpu},
                                                        encode_kwargs={"normalize_embeddings": True,
                                                                    "batch_size": 4},
                                                        query_encode_kwargs={"normalize_embeddings": True,
                                                                            "batch_size": 1})

    def acquire_location(self, latitude, longitude):
        """
        根据经纬度获取省份和城市
        """
        location = self.geolocator.reverse((latitude, longitude), language="zh")
        address = location.raw['address']
        province = address.get('state', None)
        city = address.get('city', address.get('town', address.get('county', None)))
        if province is None:
            province = ""
        if city is None:
            city = ""
        return province, city

    def acquire_device_info(self):
        """
        获取各个类型设备所有IP的相关特征信息，保存成csv
        """
        devices = self.load_all_dev_labels()

        for dev in devices:
            all_dev_filepath = list_files_in_folder(os.path.join(self.data_path, dev))
            ip_embedding_list = []
            ip_feature_list = []

            for dev_filepath in all_dev_filepath:
                
                dev_ip_count = 0

                with open(dev_filepath, "r") as f:
                    page_file = json.load(f)

                ip_results = page_file["result"]["hits"]
                
        
                for ip_result in ip_results:
                    dev_ip_count += 1

                    # 获取 IP的whois信息
                    if "whois" in ip_result.keys():
                        whois = ip_result["whois"]
                        org_name = whois["organization"]["name"] if "organization" in whois.keys() and "name" in whois["organization"].keys() else ""
                        org_handle = whois["organization"]["handle"] if "organization" in whois.keys() and "handle" in whois["organization"].keys() else ""
                        net_name = whois["network"]["name"] if "network" in whois.keys() and "name" in whois["network"].keys() else ""
                        net_handle = whois["network"]["handle"] if "network" in whois.keys() and "handle" in whois["network"].keys() else ""
                    else:
                        org_name = ""
                        org_handle = ""
                        net_name = ""
                        net_handle = ""

                    # 获取 IP的位置信息

                    if "location" in ip_result.keys():
                        location = ip_result["location"]
                        latitude = location["coordinates"]["latitude"]
                        longtitude = location["coordinates"]["longitude"]
                    else:
                        latitude = -1
                        longtitude = -1

                    # 获取 IP的反向DNS信息
                    reverse_dns_names = ""
                    if "dns" in ip_result.keys():
                        dns = ip_result["dns"]
                        if "reverse_dns" in dns.keys():
                            for index, rd in enumerate(dns["reverse_dns"]["names"]):
                                reverse_dns_names += rd
                                if index != len(dns["reverse_dns"]["names"]) - 1:
                                    reverse_dns_names += ","

                    # 获取 IP的操作系统信息
                    if "operating_system" in ip_result.keys():
                        os_cs = ip_result["operating_system"]

                        if "vendor" in os_cs.keys():
                            operating_system_vendor = os_cs["vendor"]
                        else:
                            operating_system_vendor = ""

                        if "product" in os_cs.keys():
                            operating_system_product = os_cs["product"]
                        else:
                            operating_system_product = ""

                        if "version" in os_cs.keys():
                            operating_system_version = os_cs["version"]
                        else:
                            operating_system_version = ""
                    else:
                        operating_system_vendor = ""
                        operating_system_product = ""
                        operating_system_version = ""
                    
                    # 获取 IP的服务信息
                    service_distribution = ""
                    service_names = []
                    service_ports = []
                    service_software = ""

                    http_tags = ""
                    http_favicons = ""
                    http_favicons_hash = ""
                    service_banner = ""
                    service_certificate_hash = ""
                    certificate_leaf_subject = ""
                    certificate_issuer_chain = ""
                    certificate_subject_chain = ""

                    for service in ip_result["services"]:
                        service_name = service["extended_service_name"]
                        service_port = service["port"]

                        service_names.append(service_name)
                        service_ports.append(service_port)
                        
                        if "http" in service.keys():
                            http_response = service["http"]["response"]
                            
                            # IP上所有运行HTTP 的 HTML Tags 信息汇总
                            if "html_tags" in http_response.keys():
                                http_tags += f"{service_name}-{service_port}: "

                                for index_tag, tag in enumerate(http_response["html_tags"]):
                                    http_tags += tag
                                    if index_tag != len(http_response["html_tags"]) - 1:
                                        http_tags += "\n"

                                http_tags += ","

                            # IP上所有运行HTTP 的 Favicons 信息汇总
                            if "favicons" in http_response.keys():
                                http_favicons += f"{service_name}-{service_port}: ["
                                http_favicons_hash += f"{service_name}-{service_port}: ["

                                for index_favicon, favicon in enumerate(http_response["favicons"]):

                                    http_favicons += favicon["name"]
                                    http_favicons_hash += str(favicon["shodan_hash"])
                                    
                                    if index_favicon != len(http_response) - 1:
                                        http_favicons += ","
                                        http_favicons_hash += ","
                                     
        
                                http_favicons += "],"
                                http_favicons_hash += "],"

                        # IP上所有运行服务的Banner信息汇总
                        if "banner_hex" in service.keys():
                            service_banner += f"{service_name}-{service_port}: "
                            service_banner += parse_banner_hex(service["banner_hex"])
                            service_banner += ","

                        # IP上所有运行服务的证书hash汇总
                        if "certificate" in service.keys():
                            service_certificate_hash += f"{service_name}-{service_port}: "
                            service_certificate_hash += service["certificate"]
                            service_certificate_hash += ","

                        # IP上所有运行服务的证书主题和颁发者信息汇总
                        
                        if "tls" in service.keys():
                            certificate_leaf_subject += f"{service_name}-{service_port}: "
                            certificate_issuer_chain += f"{service_name}-{service_port}: ["
                            certificate_subject_chain += f"{service_name}-{service_port}: ["

                            tls = service["tls"]
                            
                            if "certificates" in tls.keys():
                                if "leaf_data" in tls["certificates"].keys():
                                    certificate_leaf_subject += tls["certificates"]["leaf_data"]["subject_dn"]

                            if "presented_chain" in tls.keys():
                                for index_pc, pc in enumerate(tls["presented_chain"]):
                                    if "issuer_dn" in pc.keys():
                                        certificate_issuer_chain += pc["issuer_dn"]
                                    if "subject_dn" in pc.keys():
                                        certificate_subject_chain += pc["subject_dn"]

                                    if index_pc != len(tls["presented_chain"]) - 1:
                                        certificate_issuer_chain += ","
                                        certificate_subject_chain += ","
                                
                                certificate_issuer_chain += "]"
                                certificate_subject_chain += "]"
                            
                            certificate_leaf_subject += ","
                            certificate_issuer_chain += ","
                            certificate_subject_chain += ","

                        # IP上所有运行服务的软件名汇总
                        if "software" in service.keys():
                            service_software += f"{service_name}-{service_port}: ["

                            for software in service["software"]:
                                vendor_available = False
                                version_available = False
                                if "vendor" in software.keys():
                                    service_software += software["vendor"]
                                    vendor_available = True
                                if "version" in software.keys():
                                    if vendor_available:
                                        service_software += " "
                                    service_software += software["version"]
                                    version_available = True
                                if vendor_available or version_available:
                                    service_software += ","   
                            service_software += "],"    

                    # org_name_embedding = self.embedding_model.embed_query(org_name)
                    # org_handle_embedding = self.embedding_model.embed_query(org_handle)
                    # net_name_embedding = self.embedding_model.embed_query(net_name)
                    # net_handle_embedding = self.embedding_model.embed_query(net_handle)
                    # latitude_embedding = self.embedding_model.embed_query(str(latitude))
                    # longtitude_embedding = self.embedding_model.embed_query(str(longtitude))
                    # dns_embedding = self.embedding_model.embed_query(reverse_dns_names)
                    # operating_system_vendor_embedding = self.embedding_model.embed_query(operating_system_vendor)
                    # operating_system_product_embedding = self.embedding_model.embed_query(operating_system_product)
                    # operating_system_version_embedding = self.embedding_model.embed_query(operating_system_version)
                    # http_tags_embedding = self.embedding_model.embed_query(http_tags)
                    # http_favicons_embedding = self.embedding_model.embed_query(http_favicons)
                    # http_favicons_hash_embedding = self.embedding_model.embed_query(http_favicons_hash)
                    # service_banner_embedding = self.embedding_model.embed_query(service_banner)
                    # service_certificate_hash_embedding = self.embedding_model.embed_query(service_certificate_hash)
                    # certificate_leaf_subject_embedding = self.embedding_model.embed_query(certificate_leaf_subject)
                    # certificate_issuer_chain_embedding = self.embedding_model.embed_query(certificate_issuer_chain)
                    # certificate_subject_chain_embedding = self.embedding_model.embed_query(certificate_subject_chain)

                    # 根据端口升序对服务名排序并拼接
                    sorted_pairs = sorted(zip(service_ports, service_names), key=lambda x: x[0])
                    service_distribution = ",".join([f"{port}-{name}" for port, name in sorted_pairs])
                    # service_distribution_embedding = self.embedding_model.embed_query(service_distribution)
                    
                    # 构建用于聚类的所有IP的向量表示
                    # ip_embedding_list.append({
                    #     "ip": ip_result["ip"],
                    #     "org_name": org_name_embedding,
                    #     "org_handle": org_handle_embedding,
                    #     "net_name": net_name_embedding,
                    #     "net_handle": net_handle_embedding,
                    #     "latitude": latitude_embedding,
                    #     "longtitude": longtitude_embedding,
                    #     # "location": location_embedding,
                    #     "reverse_dns": dns_embedding,
                    #     "operating_system_vendor": operating_system_vendor_embedding,
                    #     "operating_system_product": operating_system_product_embedding,
                    #     "operating_system_version": operating_system_version_embedding,
                    #     "service_distribution": service_distribution_embedding,
                    #     "http_tags": http_tags_embedding,
                    #     "http_favicons": http_favicons_embedding,
                    #     "http_favicons_hash": http_favicons_hash_embedding,
                    #     "service_banner": service_banner_embedding,
                    #     "service_certificate_hash": service_certificate_hash_embedding,
                    #     "certificate_leaf_subject": certificate_leaf_subject_embedding,
                    #     "certificate_issuer_chain": certificate_issuer_chain_embedding,
                    #     "certificate_subject_chain": certificate_subject_chain_embedding
                    # })

                    # 构建所有IP的原始特征值
                    ip_feature_list.append({
                        "ip": ip_result["ip"],
                        "org_name": org_name,
                        "org_handle": org_handle,
                        "net_name": net_name,
                        "net_handle": net_handle,
                        "latitude": latitude,
                        "longtitude": longtitude,
                        # "location": province + city,
                        "reverse_dns": reverse_dns_names,
                        "operating_system_vendor": operating_system_vendor,
                        "operating_system_product": operating_system_product,
                        "operating_system_version": operating_system_version,
                        "service_distribution": service_distribution,
                        "service_software": service_software,
                        "http_tags": http_tags,
                        "http_favicons": http_favicons,
                        "http_favicons_hash": http_favicons_hash,
                        "service_banner": service_banner,
                        "service_certificate_hash": service_certificate_hash,
                        "certificate_leaf_subject": certificate_leaf_subject,
                        "certificate_issuer_chain": certificate_issuer_chain,
                        "certificate_subject_chain": certificate_subject_chain
                    })

                    logging.info("Device {} done!".format(ip_result["ip"]))

                print("{} done!".format(dev_filepath))

            ip_raw_df = pd.DataFrame(ip_feature_list, columns=self.feature_cols)
            ip_raw_df.to_csv(os.path.join(self.save_path, "ipraw_{}.csv".format(dev)), index=False, escapechar='\\', encoding='utf-8')

    def cluster_embedding(self, cols: list[str], perspective_name: str, whether_pca: bool=True):
        """
        根据embedding model获取的combination feature embedding表示, 进行聚类
        :param cols: 特征列名列表
        :param perspective_name: 视角名称
        :param whether_pca: 是否使用PCA降维
        """

        # Step 0: 设定一些初始路径
        cluster_save_path = os.path.join(self.save_path, "embedding_{}".format(perspective_name))
        os.makedirs(cluster_save_path, exist_ok=True)

        json_dir = os.path.join(cluster_save_path, "json")
        os.makedirs(json_dir, exist_ok=True)

        png_dir = os.path.join(cluster_save_path, "png")
        os.makedirs(png_dir, exist_ok=True)
        
        degrade_features = 256
        embedding_cols = [f"embedding{i+1}" for i in range(1024)]
        pca_feature_cols = [f"pca{i+1}" for i in range(degrade_features)]
        dbscan_feature_name = "cluster"

        done_dev = []

        # Step 1: 获取每个设备类型的csv，进行聚类
        for dev in self.device_labels:

            # 清理cuda内存
            if self.gpu != -1:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            src_path = os.path.join(self.save_path, f"ipraw_{dev}.csv")
            if not os.path.exists(src_path):
                logging.warning("[ERROR] CSV file not found for device %s: %s", dev, src_path)
                continue

            # Step 1-0: 设定embedding保存的目录 embedding_save_path, pca降维聚类的保存目录 pca_cluster_path, 不使用pca聚类的保存目录 nopca_cluster_path
            embedding_save_path = os.path.join(cluster_save_path, f"ipraw_{dev}_embedding_{perspective_name}.csv")
            pca_cluster_path = os.path.join(cluster_save_path, f"ipraw_{dev}_embedding_{perspective_name}_pca.csv")
            nopca_cluster_path = os.path.join(cluster_save_path, f"ipraw_{dev}_embedding_{perspective_name}_cluster.csv")

            if whether_pca and os.path.exists(pca_cluster_path):
                # print(f"[{perspective_name} Embedding (PCA: {whether_pca})] Embedding for Device {dev} has been done!")
                logging.info(f"[{perspective_name} Embedding (PCA: {whether_pca})] Embedding for Device {dev} has been done!")
                done_dev.append(dev)
                continue

            if not whether_pca and os.path.exists(nopca_cluster_path):
                # print(f"[{perspective_name} Embedding (PCA: {whether_pca})] Embedding for Device {dev} has been done!")
                logging.info(f"[{perspective_name} Embedding (PCA: {whether_pca})] Embedding for Device {dev} has been done!")
                done_dev.append(dev)
                continue    

            # Step 1-1: 获取一个perspective角度下的特征embedding dataframe。如果已经存在，则直接读取。否则，使用embedding model获取。

            # print(f"[{perspective_name} Embedding (PCA: {whether_pca})] Embedding for Device {dev} running...")
            logging.info(f"[{perspective_name} Embedding (PCA: {whether_pca})] Embedding for Device {dev} running...")
            
            if os.path.exists(embedding_save_path):
                embedding_df = pd.read_csv(embedding_save_path)
                # print(f"[{perspective_name} Embedding (PCA: {whether_pca})] Embedding for Device {dev} loaded!")
                logging.info(f"[{perspective_name} Embedding (PCA: {whether_pca})] Embedding for Device {dev} loaded!")
                
            else:
                df = pd.read_csv(src_path)
                
                missing_cols = [c for c in cols if c not in df.columns]
                if missing_cols:
                    logging.warning(
                        "Device %s CSV missing %s columns: %s, skip.",
                        dev,
                        len(missing_cols),
                        missing_cols,
                    )
                    continue

                aggregation_col = f"{perspective_name}_combination"
                
                # 1-1-1：将所有特征列拼接成一个字符串
                df[aggregation_col] = df[cols].astype(str).agg("|".join, axis=1)

                # 1-1-2：获取embedding
                embeddings = []

                # 查找最后的milestone文件
                milestone_files = [f for f in os.listdir(cluster_save_path) if f.startswith(f"ipraw_{dev}_embedding_{perspective_name}_milestone_")]
                if milestone_files:
                    # 提取milestone文件中的index并找到最大值
                    milestone_indices = [int(f.split('_')[-1].replace('.csv', '')) for f in milestone_files]
                    last_milestone_index = max(milestone_indices)
    
                    # 从milestone文件加载已有的embeddings
                    last_milestone_path = os.path.join(cluster_save_path, f"ipraw_{dev}_embedding_{perspective_name}_milestone_{last_milestone_index}.csv")
                    milestone_df = pd.read_csv(last_milestone_path)
                    embeddings = milestone_df[embedding_cols].values.tolist()
        
                    # print(f"[{perspective_name} Embedding (PCA: {whether_pca})] Resuming from milestone at index {last_milestone_index} for Device {dev}")
                    logging.info(f"[{perspective_name} Embedding (PCA: {whether_pca})] Resuming from milestone at index {last_milestone_index} for Device {dev}")
    
                    start_index = last_milestone_index
                else:
                    start_index = 0
                    embeddings = []

                for index, text in enumerate(df[aggregation_col].values[start_index:], start=start_index):
                    
                    try_time = 0
                    success_flag = False
                    while try_time < 3:
                        try:
                            emb = self.embedding_model.embed_query(str(text))
                            embeddings.append(emb)
                            success_flag = True
                            break
                        except torch.OutOfMemoryError as e:
                            if self.gpu != -1:
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats()
                            try_time += 1
                            logging.error(e)
                            continue
                    
                    if not success_flag:
                        logging.info("[DEBUG] Chunking Text index: {}".format(index))
                        chunks = chunk_text(str(text))
                        try:
                            embs = self.embedding_model.embed_documents(chunks)
                            final_emb = numpy.mean(embs, axis=0)
                            logging.info("[DEBUG] Embedding Length: {}".format(len(final_emb)))
                            embeddings.append(final_emb)
                        except torch.OutOfMemoryError as e:
                            if self.gpu != -1:
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats()
                            logging.error(e)
                            raise Exception(f"[{perspective_name} Embedding (PCA: {whether_pca})] Embedding for Device {dev} failed at index {index}")
                    
                    if index % 10000 == 0 and index > 0:
                        # print(f"[{perspective_name} Embedding (PCA: {whether_pca})] Embedding for Device {dev} running... {index}/{len(df)}")
                        logging.info(f"[{perspective_name} Embedding (PCA: {whether_pca})] Embedding for Device {dev} running... {index}/{len(df)}")

                        milestone_df = pd.DataFrame(embeddings[:index], columns=embedding_cols)
                        milestone_df.insert(0, "ip", df["ip"].values[:index])
                        milestone_path = os.path.join(cluster_save_path, f"ipraw_{dev}_embedding_{perspective_name}_milestone_{index}.csv")
                        milestone_df.to_csv(milestone_path, index=False)
                        
                        # print(f"[{perspective_name} Embedding (PCA: {whether_pca})] Saved milestone at {index} for Device {dev}")
                        logging.info(f"[{perspective_name} Embedding (PCA: {whether_pca})] Saved milestone at {index} for Device {dev}")
                
                logging.info(f"Embedding Length: {len(embeddings)}")
                
                # 2.3: 创建新的DataFrame保存IP和embedding, 并保存到daraframe中
                embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
                embedding_df.insert(0, "ip", df["ip"].values)
                embedding_df.to_csv(embedding_save_path, index=False)

                # 将所有milestone文件移到milestone文件夹下
                milestone_dir = os.path.join(cluster_save_path, "milestone")
                os.makedirs(milestone_dir, exist_ok=True)

                milestone_files = [f for f in os.listdir(cluster_save_path) if f.startswith(f"ipraw_{dev}_embedding_{perspective_name}_milestone_")]
                for milestone_file in milestone_files:
                    src = os.path.join(cluster_save_path, milestone_file)
                    dst = os.path.join(milestone_dir, milestone_file)
                    shutil.move(src, dst)

            # print("[**] Qwen3 embedding done!")
            logging.info("[**] Qwen3 embedding done!")

            if whether_pca:
                # Step 2.4: PCA降维

                logging.info(f"[PCA] PCA for Device {dev} running...")
                pca = PCA(n_components=degrade_features)
                X_pca = pca.fit_transform(embedding_df[embedding_cols].values)

                pca_df = pd.DataFrame(X_pca)
                pca_df.columns = pca_feature_cols

                pca_df.insert(0, "ip", embedding_df["ip"].values)
                # print(f"[PCA] PCA for Device {dev} done!")
                logging.info(f"[PCA] PCA for Device {dev} done!")

                # Step 3: HDBSCAN 聚类
                # print(f"[HDBSCAN] HDBSCAN for Device {dev} running...")
                logging.info(f"[HDBSCAN] HDBSCAN for Device {dev} running...")

                X = pca_df[pca_feature_cols].values
                dbscan = hdbscan.HDBSCAN(min_cluster_size=20)
                predict_clusters = dbscan.fit_predict(X)

                if (predict_clusters == -1).all():
                    predict_clusters = 0
                
                pca_df[dbscan_feature_name] = predict_clusters

                # Step 4: 统计HDBSCAN聚类结果
                unique_dbscan_clusters = pca_df[dbscan_feature_name].unique()
                n_dbscan_clusters = len(unique_dbscan_clusters[unique_dbscan_clusters != -1])
                n_noise = (pca_df[dbscan_feature_name] == -1).sum()

                # print(f"[HDBSCAN] Cluster Result of Device {dev}: {n_dbscan_clusters} clusters, {n_noise} noise points")
                logging.info(f"[HDBSCAN] Cluster Result of Device {dev}: {n_dbscan_clusters} clusters, {n_noise} noise points")
                
                # print(f"[HDBSCAN] Clustering done for Device {dev}")
                logging.info(f"[HDBSCAN] Clustering done for Device {dev}")

                # Step 5: 保存结果
                pca_df.to_csv(pca_cluster_path, index=False)
                # print(f"PCA (256) Embedding & Clustering Result Saved at {pca_cluster_path}")
                logging.info(f"PCA (256) Embedding & Clustering Result Saved at {pca_cluster_path}")

            else:
                # Step 3: HDBSCAN 聚类
                # print(f"[HDBSCAN] HDBSCAN for Device {dev} running...")
                logging.info(f"[HDBSCAN] HDBSCAN for Device {dev} running...")

                X = embedding_df[embedding_cols].values
                dbscan = hdbscan.HDBSCAN(min_cluster_size=20)
                predict_clusters = dbscan.fit_predict(X)

                if (predict_clusters == -1).all():
                    predict_clusters = 0
                
                embedding_df[dbscan_feature_name] = predict_clusters

                # Step 4: 统计HDBSCAN聚类结果
                unique_dbscan_clusters = embedding_df[dbscan_feature_name].unique()
                n_dbscan_clusters = len(unique_dbscan_clusters[unique_dbscan_clusters != -1])
                n_noise = (embedding_df[dbscan_feature_name] == -1).sum()

                # print(f"[HDBSCAN] Cluster Result of Device {dev}: {n_dbscan_clusters} clusters, {n_noise} noise points")
                logging.info(f"[HDBSCAN] Cluster Result of Device {dev}: {n_dbscan_clusters} clusters, {n_noise} noise points")

                # print(f"[HDBSCAN] Clustering done for Device {dev}")
                logging.info(f"[HDBSCAN] Clustering done for Device {dev}")
                
                # Step 5: 保存结果
                embedding_df.to_csv(nopca_cluster_path, index=False)

                # print(f"NoPCA (1024) Embedding & Clustering Result Saved at {nopca_cluster_path}")
                logging.info(f"NoPCA (1024) Embedding & Clustering Result Saved at {nopca_cluster_path}")
        
        # 如果所有dev已经embedding完毕，跳过画图阶段
        if len(done_dev) == len(self.device_labels):
            logging.info(f"All devices have been embedded for perspective {perspective_name}")
            return True
        
        if whether_pca:
            # Step 6: 绘制聚类结果
            self.draw_cluster_2D(pca_feature_cols[:2], f"embedding_{perspective_name}", "dbscan", whether_pca=True, whether_embedding=True)
        else:
            # Step 6: 绘制聚类结果
            self.draw_cluster_2D(embedding_cols[:2], f"embedding_{perspective_name}", "dbscan", whether_pca=False, whether_embedding=True)
        
        return False

    def overall_cluster_embedding(self, contact_done: bool = False):
        overall_dir = os.path.join(self.save_path, "embedding_overall")
        if not contact_done:
            # 创建保存目录
            os.makedirs(overall_dir, exist_ok=True)
            
            for dev in self.device_labels:
                overall_df = None
                for perspective_name in self.perspective_names:
                    embedding_file_path = os.path.join(self.save_path, f"embedding_{perspective_name}/ipraw_{dev}_embedding_{perspective_name}_pca.csv")
                    df = pd.read_csv(embedding_file_path)
                    rename_dict = {f"pca{i}": f"{perspective_name}{i}" for i in range(1, 257)}
                    df = df.rename(columns=rename_dict)
                    
                    renamed_cols = [f"{perspective_name}{i}" for i in range(1, 257)]

                    if overall_df is None:
                        overall_df = df.iloc[:, :-1]
                    else:
                        df = df[renamed_cols]

                        overall_df = pd.concat([overall_df, df], axis=1)

                    print(f"{dev}-{perspective_name}: {len(overall_df.columns)}")
                
                output_path = os.path.join(overall_dir, f"ipraw_{dev}_embedding_overall_raw.csv")
                overall_df.to_csv(output_path, index=False)
                logging.info("Saved overall embedding for device %s to %s", dev, output_path)

        for dev in self.device_labels:

            overall_df = pd.read_csv(os.path.join(overall_dir, f"ipraw_{dev}_embedding_pca.csv"))

            # 聚类
            feature_cols = [col for col in overall_df.columns if col != "ip"]
            X = overall_df[feature_cols].values

            # 使用HDBScan进行自适应聚类
            clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
            cluster_labels = clusterer.fit_predict(X)

            # 将聚类结果添加到dataframe
            overall_df['cluster'] = cluster_labels

            # 保存聚类结果
            output_path = os.path.join(overall_dir, f"ipraw_{dev}_embedding_overall_pca.csv")
            overall_df.to_csv(output_path, index=False)
            logging.info("Saved overall embedding for device %s to %s", dev, output_path)

    def concat_perspective_embeddings(self):
        """
        将每个设备dev的各个embedding_{perspective_name}目录下的
        ipraw_{dev}_embedding_{perspective_name}.csv 进行特征列的横向拼接，
        拼接后保存到 embedding_local/ipraw_{dev}_embedding.csv
        """
        output_dir = os.path.join(self.save_path, "embedding_local")
        os.makedirs(output_dir, exist_ok=True)
        
        for dev in self.device_labels:
            dfs = []
            ip_col = None

            for perspective_name in self.local_retrieval_perspective_names:
                csv_path = os.path.join(self.save_path, f"embedding_{perspective_name}", f"ipraw_{dev}_embedding_{perspective_name}.csv")

                if not os.path.exists(csv_path):
                    logging.warning("Embedding CSV not found, skip: %s", csv_path)
                    continue

                df = pd.read_csv(csv_path)

                # 保留第一个文件的ip列
                if ip_col is None:
                    ip_col = df[["ip"]]

                # 取出embedding特征列（排除ip列），并加上perspective前缀避免列名冲突
                feature_cols = [c for c in df.columns if c != "ip"]
                rename_dict = {c: f"{perspective_name}{index + 1}" for index, c in enumerate(feature_cols)}
                # print(rename_dict)
                dfs.append(df[feature_cols].rename(columns=rename_dict))

            if ip_col is None or len(dfs) == 0:
                logging.warning("No embedding CSV found for device %s, skip.", dev)
                continue

            merged_df = pd.concat([ip_col] + dfs, axis=1)
            output_path = os.path.join(output_dir, f"ipraw_{dev}_embedding.csv")
            merged_df.to_csv(output_path, index=False)
            logging.info("Saved concatenated embedding for device %s to %s (%d cols)", dev, output_path, len(merged_df.columns))

    def HGT_cluster_embedding(self, pca_components: int = 256):
        """
        根据 HGT.py 生成的设备全局表示结果，对各类设备进行聚类。
        读取 hgt_embeddings/device_hgt_embeddings.csv，按设备类型拆分后执行
        HDBSCAN 聚类，结果保存至 community/embedding_overall/ipraw_{dev}_embedding_overall.csv。

        :param pca_components: PCA降维目标维度（0 表示不降维，直接在1024维上聚类）
        """
        hgt_emb_path = os.path.join(self.save_path, "hgt_embeddings", "device_hgt_embeddings.csv")
        if not os.path.exists(hgt_emb_path):
            logging.error("[HGT_cluster] HGT embedding file not found: %s", hgt_emb_path)
            print(f"[HGT_cluster] HGT embedding file not found: {hgt_emb_path}")
            return

        output_dir = os.path.join(self.save_path, "community", "embedding_overall")
        os.makedirs(output_dir, exist_ok=True)

        print("[HGT_cluster] Loading HGT device embeddings...")
        logging.info("[HGT_cluster] Loading HGT device embeddings...")
        hgt_df = pd.read_csv(hgt_emb_path)
        hgt_emb_cols = [c for c in hgt_df.columns if c != "ip"]

        for dev in self.device_labels:
            ipraw_path = os.path.join(self.save_path, f"ipraw_{dev}.csv")
            if not os.path.exists(ipraw_path):
                logging.warning("[HGT_cluster] ipraw CSV not found for device %s: %s", dev, ipraw_path)
                continue

            dev_ips = pd.read_csv(ipraw_path, usecols=["ip"])["ip"]
            dev_df = hgt_df[hgt_df["ip"].isin(dev_ips)].reset_index(drop=True)

            if dev_df.empty:
                logging.warning("[HGT_cluster] No HGT embeddings found for device %s, skip.", dev)
                continue

            print(f"[HGT_cluster] Device {dev}: {len(dev_df)} IPs")
            logging.info("[HGT_cluster] Device %s: %d IPs", dev, len(dev_df))

            X = dev_df[hgt_emb_cols].values

            # 可选PCA降维
            apply_pca = pca_components > 0 and X.shape[1] > pca_components
            if apply_pca:
                logging.info("[HGT_cluster] Applying PCA (%d → %d) for device %s", X.shape[1], pca_components, dev)
                pca = PCA(n_components=pca_components)
                X = pca.fit_transform(X)

            # 根据是否PCA决定输出文件名
            output_filename = (
                f"ipraw_{dev}_embedding_overall_pca.csv" if apply_pca
                else f"ipraw_{dev}_embedding_overall.csv"
            )
            output_path = os.path.join(output_dir, output_filename)

            # HDBSCAN聚类
            logging.info("[HGT_cluster] HDBSCAN clustering for device %s...", dev)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
            cluster_labels = clusterer.fit_predict(X)

            if (cluster_labels == -1).all():
                cluster_labels[:] = 0

            unique_clusters = len(set(cluster_labels) - {-1})
            n_noise = (cluster_labels == -1).sum()
            print(f"[HGT_cluster] Device {dev}: {unique_clusters} clusters, {n_noise} noise points")
            logging.info("[HGT_cluster] Device %s: %d clusters, %d noise points", dev, unique_clusters, n_noise)

            result_df = dev_df.copy()
            result_df["cluster"] = cluster_labels
            result_df.to_csv(output_path, index=False)
            logging.info("[HGT_cluster] Saved to %s", output_path)

        print("[HGT_cluster] Done.")
        logging.info("[HGT_cluster] Done.")

    def cluster_perspective(
        self,
        n_clusters: int = 20,
        whether_aggregated: bool = False,
        cols: list[str] | None = None,
        perspective_name: str = "os",
    ):
        """
        从已有的 ipraw_*.csv 中读取指定特征列，执行聚类，并输出结果文件。

        :param n_clusters: 聚类数量上限
        :param whether_aggregated: True 时仅对所有特征拼接后的列聚类，False 时分别编码各列聚类
        :param cols: 需要聚类的特征列列表，默认为操作系统相关列
        :param perspective_name: 聚类视角名称，用于文件名
        :param src_pattern: 数据源文件相对路径模式，默认 ipraw_{dev}.csv
        """

        cluster_save_path = os.path.join(self.save_path, perspective_name)
        os.makedirs(cluster_save_path, exist_ok=True)
        json_dir = os.path.join(cluster_save_path, "json")
        os.makedirs(json_dir, exist_ok=True)
        png_dir = os.path.join(cluster_save_path, "png")
        os.makedirs(png_dir, exist_ok=True)

        if cols is None:
            cols = [
                "operating_system_vendor",
                "operating_system_product",
                "operating_system_version",
            ]

        for dev in self.device_labels:
            src_path = os.path.join(self.save_path, f"ipraw_{dev}.csv")
            if not os.path.exists(src_path):
                logging.warning("CSV file not found for device %s: %s", dev, src_path)
                continue

            logging.info("Start %s clustering for device %s from %s", perspective_name, dev, src_path)

            df = pd.read_csv(src_path)

            missing_cols = [c for c in cols if c not in df.columns]
            if missing_cols:
                logging.warning(
                    "Device %s CSV missing %s columns: %s, skip.",
                    dev,
                    perspective_name,
                    ",".join(missing_cols),
                )
                continue

            df[cols] = df[cols].fillna("")

            code_cols = []
            factorize_mapping = {}

            if whether_aggregated:
                aggregation_col = f"{perspective_name}_combination"
                df[aggregation_col] = df[cols].astype(str).agg("|".join, axis=1)

                codes, uniques = pd.factorize(df[aggregation_col], sort=True)
                code_col = f"{aggregation_col}_code"
                df[code_col] = codes.astype("int32")
                code_cols.append(code_col)
                factorize_mapping[aggregation_col] = {
                    int(code): str(value) for code, value in enumerate(uniques)
                }
            else:
                new_df = df[cols]

                for col in cols:
                    codes, uniques = pd.factorize(new_df[col], sort=True)
                    code_col = f"{col}_code"
                    df[code_col] = codes.astype("int32")
                    code_cols.append(code_col)

                    # 保存映射关系：code -> value
                    factorize_mapping[col] = {
                        int(code): str(value) for code, value in enumerate(uniques)
                    }

            if len(df) == 0:
                logging.warning("No rows in CSV for device %s, skip.", dev)
                continue

            cluster_feature_name = f"{perspective_name}_cluster"
            dbscan_feature_name = f"{perspective_name}_dbscan_cluster"

            # 如果样本数量小于设定聚类数，则降低聚类数
            k = min(max(n_clusters, 1), len(df))
            if k <= 1:
                # 所有样本都分到同一簇
                df[cluster_feature_name] = 0
            else:
                X = df[code_cols].values

                # 指定聚类数量：使用KMeans
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                df[cluster_feature_name] = km.fit_predict(X)

                logging.info("[KMeans] Clustering done for device %s", dev)

                # 自动聚类方法：使用HDBSCAN，无需指定聚类数量
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # 使用DBSCAN进行自动聚类
                # dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan = hdbscan.HDBSCAN(min_cluster_size=20)
                df[dbscan_feature_name] = dbscan.fit_predict(X_scaled)

                # 如果所有点都被标记为噪声(-1)，则将它们归为一类
                if (df[dbscan_feature_name] == -1).all():
                    df[dbscan_feature_name] = 0
                
                # 统计DBSCAN聚类结果
                unique_dbscan_clusters = df[dbscan_feature_name].unique()
                n_dbscan_clusters = len(unique_dbscan_clusters[unique_dbscan_clusters != -1])
                n_noise = (df[dbscan_feature_name] == -1).sum()
                logging.info("[HDBSCAN] Device %s: %d clusters, %d noise points", dev, n_dbscan_clusters, n_noise)
                print(f"[HDBSCAN] Device {dev}: {n_dbscan_clusters} clusters, {n_noise} noise points")
                
                logging.info("[HDBSCAN] Clustering done for device %s", dev)

            if whether_aggregated:
                dst_path = os.path.join(cluster_save_path, f"aggregated/ipraw_{dev}_{perspective_name}_cluster_aggregated.csv")
            else:
                dst_path = os.path.join(cluster_save_path, f"ipraw_{dev}_{perspective_name}_cluster.csv")

            df.to_csv(dst_path, index=False)
            
            # 保存 factorize 映射关系到 JSON
            if whether_aggregated:
                mapping_path = os.path.join(json_dir, f"{perspective_name}_code_mapping_{dev}_aggregated.json")
            else:
                mapping_path = os.path.join(json_dir, f"{perspective_name}_code_mapping_{dev}.json")
            
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(factorize_mapping, f, ensure_ascii=False, indent=2)
            
            logging.info(
                "[*] Finished %s clustering for device %s, result saved to %s", perspective_name, dev, dst_path
            )
            logging.info("[+] Code mapping saved to: %s", mapping_path)
    
    def draw_cluster_2D(self, cols: list[str], perspective_name: str, clusterer: str = "kmeans", whether_pca: bool = False, whether_embedding: bool = False):
        """
        绘制两个特征的二维聚类空间图
        从 iot-classification/csv/{perspective_name}中读取聚类后的文件
        """

        if len(cols) != 2:
            raise ValueError("draw_cluster_2D requires exactly two column names in 'cols'.")

        cluster_dir = os.path.join(self.save_path, perspective_name)
        png_dir = os.path.join(cluster_dir, "png")

        os.makedirs(cluster_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        if clusterer == "kmeans":
            cluster_col = f"{perspective_name}_cluster"
        elif clusterer == "dbscan":
            cluster_col = f"{perspective_name}_dbscan_cluster"
        else:
            cluster_col = f"{perspective_name}_{clusterer}_cluster"

        if whether_embedding:
            cluster_col = "cluster"

        for dev in self.device_labels:
            if whether_pca:
                cluster_csv_path = os.path.join(cluster_dir, f"ipraw_{dev}_{perspective_name}_pca.csv")
                required_cols = [
                    cols[0],
                    cols[1],
                    cluster_col
                ]
            elif whether_embedding:
                cluster_csv_path = os.path.join(cluster_dir, f"ipraw_{dev}_{perspective_name}_cluster.csv")
                required_cols = [
                    cols[0],
                    cols[1],
                    cluster_col
                ]
            else:
                cluster_csv_path = os.path.join(cluster_dir, f"ipraw_{dev}_{perspective_name}_cluster.csv")
                required_cols = [
                    "{}_code".format(cols[0]),
                    "{}_code".format(cols[1]),
                    cluster_col
                ]

            if not os.path.exists(cluster_csv_path):
                logging.warning("Cluster CSV not found for device %s: %s", dev, cluster_csv_path)
                print(f"Cluster CSV not found for {dev}, skip.")
                continue
            
            logging.info("[*] Drawing 2D cluster plot for device %s - perspective %s", dev, perspective_name)
            # print(f"[*] Loading 2D cluster data for {dev}...")
            
            df = pd.read_csv(cluster_csv_path)
            
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logging.warning("Device %s missing columns: %s, skip.", dev, missing)
                print(f"Missing columns for {dev}: {missing}, skip.")
                continue
            
            if df.empty:
                logging.warning("No rows for device %s in %s", dev, cluster_csv_path)
                continue

            x = pd.to_numeric(df[required_cols[0]], errors="coerce").fillna(0.0).to_numpy()
            y = pd.to_numeric(df[required_cols[1]], errors="coerce").fillna(0.0).to_numpy()
            clusters = df[cluster_col].to_numpy()

            unique_clusters = numpy.unique(clusters)
            if len(unique_clusters) == 0:
                logging.warning("No cluster labels found for device %s", dev)
                continue

            centers = {}
            for label in unique_clusters:
                mask = clusters == label
                centers[label] = numpy.array([x[mask].mean(), y[mask].mean()]) if mask.any() else numpy.array([0.0, 0.0])

            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            pad_x = max((x_max - x_min) * 0.05, 1e-6)
            pad_y = max((y_max - y_min) * 0.05, 1e-6)

            xx, yy = numpy.meshgrid(
                numpy.linspace(x_min - pad_x, x_max + pad_x, 400),
                numpy.linspace(y_min - pad_y, y_max + pad_y, 400),
            )

            grid_points = numpy.c_[xx.ravel(), yy.ravel()]
            center_array = numpy.array([centers[label] for label in unique_clusters])

            diff = grid_points[:, None, :] - center_array[None, :, :]
            distances = numpy.sum(diff ** 2, axis=2)
            nearest_idx = numpy.argmin(distances, axis=1)
            Z = nearest_idx.reshape(xx.shape)

            cmap = plt.get_cmap("tab20", max(len(unique_clusters), 1))
            contour_levels = numpy.arange(-0.5, len(unique_clusters) + 0.5, 1)

            fig, ax = plt.subplots(figsize=(30, 29))
            ax.contourf(xx, yy, Z, levels=contour_levels, cmap=cmap, alpha=0.2)

            for idx, label in enumerate(unique_clusters):
                mask = clusters == label
                color = cmap(idx)
                ax.scatter(
                    x[mask],
                    y[mask],
                    c=[color],
                    label=f"Cluster {label}",
                    s=35,
                    edgecolors="k",
                    linewidths=0.3,
                    alpha=0.85,
                )

                center = centers[label]
                ax.scatter(
                    center[0],
                    center[1],
                    c=[color],
                    marker="X",
                    s=150,
                    edgecolors="black",
                    linewidths=0.8,
                    label=f"Center {label}",
                )

            ax.set_xlabel(required_cols[0])
            ax.set_ylabel(required_cols[1])
            ax.set_title(f"2D {perspective_name.upper()} Clustering for {dev.upper()} (Count: {len(df)})", fontsize=14, fontweight="bold")

            if not whether_embedding:
                if clusterer == "kmeans":
                    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.5)
                else:
                    ax.legend(loc="best", fontsize=13, ncol=15, framealpha=0.5)

            ax.grid(True, linestyle="--", alpha=0.3)

            plt.tight_layout()

            if "kmeans" in clusterer:
                if whether_pca:
                    output_path = os.path.join(png_dir, f"cluster_2d_{dev}_{cols[0]}_{cols[1]}_kmeans(pca).png")
                else:
                    output_path = os.path.join(png_dir, f"cluster_2d_{dev}_{cols[0]}_{cols[1]}_kmeans.png")
            elif "dbscan" in clusterer:
                if whether_pca:
                    output_path = os.path.join(png_dir, f"cluster_2d_{dev}_{cols[0]}_{cols[1]}_dbscan(pca).png")
                else:
                    output_path = os.path.join(png_dir, f"cluster_2d_{dev}_{cols[0]}_{cols[1]}_dbscan.png")
            elif "gmm" in clusterer:
                if whether_pca:
                    output_path = os.path.join(png_dir, f"cluster_2d_{dev}_{cols[0]}_{cols[1]}_gmm(pca).png")
                else:
                    output_path = os.path.join(png_dir, f"cluster_2d_{dev}_{cols[0]}_{cols[1]}_gmm.png")

            plt.savefig(output_path, dpi=160, bbox_inches='tight')
            # print(f"[+] 2D cluster plot saved to: {output_path}")
            logging.info("[+] 2D cluster plot saved to: %s", output_path)

            plt.close(fig)

    def draw_cluster_3D(self, cols: list[str], perspective_name: str, clusterer: str = "kmeans"):
        """
        绘制三个特征的三维聚类空间图
        params perspective_name: embedding_{input_perspective_name}
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # perspective
        cluster_read_path = os.path.join(self.save_path, perspective_name)

        for dev in self.device_labels:
            cluster_csv_path = os.path.join(cluster_read_path, f"ipraw_{dev}_{perspective_name}_cluster.csv")
            
            if not os.path.exists(cluster_csv_path):
                logging.warning("Cluster CSV not found for device %s: %s", dev, cluster_csv_path)
                print(f"Cluster CSV not found for {dev}, skip.")
                continue
            
            logging.info("Drawing 3D cluster plot for device %s", dev)
            print(f"Loading cluster data for {dev}...")
            
            df = pd.read_csv(cluster_csv_path)
            
            if clusterer == "kmeans":
                cluster_col = f"{perspective_name}_cluster"
            elif clusterer == "dbscan":
                cluster_col = f"{perspective_name}_dbscan_cluster"
            else:
                raise ValueError("Invalid clusterer: %s", clusterer)
            
            # 检查必需的列
            required_cols = [
                "{}_code".format(cols[0]),
                "{}_code".format(cols[1]),
                "{}_code".format(cols[2]),
                cluster_col
            ]
            
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logging.warning("Device %s missing columns: %s, skip.", dev, missing)
                print(f"Missing columns for {dev}: {missing}, skip.")
                continue
            
            # 提取三维坐标和聚类标签
            X = df[required_cols[0]].values
            Y = df[required_cols[1]].values
            Z = df[required_cols[2]].values
            clusters = df[required_cols[3]].values
            
            # 创建三维图
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # 为每个聚类分配颜色
            unique_clusters = numpy.unique(clusters)
            colors = plt.cm.tab20(numpy.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster_id in enumerate(unique_clusters):
                mask = clusters == cluster_id
                ax.scatter(
                    X[mask], Y[mask], Z[mask],
                    c=[colors[i]],
                    label=f"Cluster {cluster_id}",
                    alpha=0.6,
                    s=20,
                    edgecolors='k',
                    linewidths=0.3
                )
            
            ax.set_xlabel(required_cols[0], fontsize=10)
            ax.set_ylabel(required_cols[1], fontsize=10)
            ax.set_zlabel(required_cols[2], fontsize=10)
            ax.set_title(f"3D {perspective_name} Clustering for {dev.upper()}", fontsize=14, fontweight='bold')
            
            # 图例设置
            if len(unique_clusters) <= 20:
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8, ncol=1)
            else:
                # 聚类数太多时不显示图例
                pass
            
            plt.tight_layout()
            
            # 保存图片
            png_save_path = os.path.join(self.save_path, perspective_name, f"png/cluster_3d_{dev}.png")
            plt.savefig(png_save_path, dpi=150, bbox_inches='tight')

            print(f"3D cluster plot saved to: {png_save_path}")
            logging.info("3D cluster plot saved to: %s", png_save_path)
            
            # plt.show()
            plt.close()

    def dev_distribution(self, cols: list[str], perspective_name: str):
        """
        统计每个设备中特定列的分布情况
        """
        perspective_name = f"embedding_{perspective_name}"
        logging.info(self.device_labels)

        for dev in self.device_labels:
            cluster_csv_path = os.path.join(self.save_path, f"ipraw_{dev}.csv")
            
            if not os.path.exists(cluster_csv_path):
                logging.warning("Cluster CSV not found for device %s: %s", dev, cluster_csv_path)
                print(f"Cluster CSV not found for {dev}, skip.")
                continue
            
            logging.info("Analyzing %s distribution for device %s", perspective_name, dev)
            # print(f"Analyzing {perspective_name} distribution for {dev}...")
            
            df = pd.read_csv(cluster_csv_path)
            
            # 检查必需的列
            missing = [c for c in cols if c not in df.columns]
            if missing:
                logging.warning("Device %s missing %s columns: %s, skip.", dev, perspective_name, missing)
                print(f"Missing columns for {dev}: {missing}, skip.")
                continue
            
            # 填充缺失值
            df[cols] = df[cols].fillna("")
            
            # 统计结果字典
            distribution_stats = {}
            
            # 根据 cols 的长度进行统计
            total_count = len(df)

            for col in cols:
                col_counts = df[col].value_counts().to_dict()
                distribution_stats[col] = {
                    "total_count": total_count,
                    "unique_count": len(col_counts),
                    "distribution": col_counts,
                }
            
            if len(cols) > 1:
                # 统计各列组合的分布（支持任意长度的 cols）
                combination_feature = f"{perspective_name}_combination"
                df[combination_feature] = df[cols].astype(str).agg("|".join, axis=1)
                
                combination_counts = df[combination_feature].value_counts().to_dict()
                
                # 将组合字符串转换为结构化格式
                combination_structured = {}
                for combo_str, count in combination_counts.items():
                    parts = combo_str.split("|")
                    
                    if len(parts) > len(cols):
                        new_parts = ["|".join(parts[i:i + len(cols)]) for i in range(0, len(parts), len(cols))]
                        parts = new_parts

                    combination_structured[combo_str] = {
                        "values": {
                            cols[i]: parts[i] if i < len(parts) else ""
                            for i in range(len(cols))
                        },
                        "count": int(count),
                    }
                
                distribution_stats[combination_feature] = {
                    "total_count": len(df),
                    "unique_count": len(combination_counts),
                    "distribution": combination_structured
                }

                logging.info(f"  - Unique {combination_feature}: {distribution_stats[combination_feature]['unique_count']}")

            # 保存为 JSON 文件
            output_path = os.path.join(self.save_path, perspective_name, f"json/{perspective_name}_distribution_{dev}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(distribution_stats, f, ensure_ascii=False, indent=2)
            
            # print(f"{perspective_name} distribution stats saved to: {output_path}")
            logging.info(f"{perspective_name} distribution stats saved to: %s", output_path)
            
            # 打印简要统计信息
            for col in cols:
                logging.info(f"  - Unique {col}: {distribution_stats[col]['unique_count']}")
            
            # 绘制 combination 条形图
            logging.info(f"Drawing {perspective_name} combination bar chart for {dev}...")
            
            # 获取前 30 个最常见的组合（避免图表过于拥挤）
            top_n = 30
            if len(cols) == 1:
                combination_counts = distribution_stats[cols[0]]["distribution"]

            sorted_combinations = sorted(
                combination_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
            if len(sorted_combinations) > 0:
                combo_labels = [combo[0] for combo in sorted_combinations]
                combo_values = [combo[1] for combo in sorted_combinations]
                
                # 创建条形图
                fig, ax = plt.subplots(figsize=(16, 10))
                
                # 绘制水平条形图（更适合长标签）
                y_pos = numpy.arange(len(combo_labels))
                bars = ax.barh(y_pos, combo_values, color='steelblue', alpha=0.7, edgecolor='black')
                
                # 设置标签
                ax.set_yticks(y_pos)
                ax.set_yticklabels(combo_labels, fontsize=8)
                ax.set_xlabel('Count', fontsize=12, fontweight='bold')
                if perspective_name == "os":
                    ax.set_ylabel(f'{perspective_name} Combination (Vendor|Product|Version)', fontsize=12, fontweight='bold')
                elif perspective_name == "whois":
                    ax.set_ylabel(f'{perspective_name} Combination (Org|Net)', fontsize=12, fontweight='bold')
                ax.set_title(f'Top {len(sorted_combinations)} {perspective_name} Combinations for {dev.upper()}', 
                            fontsize=14, fontweight='bold', pad=20)
                
                # 在条形上添加数值标签
                for i, (bar, value) in enumerate(zip(bars, combo_values)):
                    ax.text(value, i, f' {value}', va='center', fontsize=8)
                
                # 反转 y 轴，使最高的在顶部
                ax.invert_yaxis()
                
                plt.tight_layout()
                
                # 保存图片
                bar_chart_path = os.path.join(self.save_path, perspective_name, f"png/{perspective_name}_combination_bar_{dev}.png")
                plt.savefig(bar_chart_path, dpi=150, bbox_inches='tight')

                # print(f"{perspective_name} combination bar chart saved to: {bar_chart_path}")
                logging.info("%s combination bar chart saved to: %s", perspective_name, bar_chart_path)
                
                plt.close()
            else:
                logging.error(f"No {perspective_name} combinations to plot for {dev}")

    def dev_whois_analysis(self):
        """
        分析设备的whois信息, 增加whois_org和whois_net列
        """
        perspective_name = "whois"
        whois_dir = os.path.join(self.save_path, perspective_name)
        os.makedirs(whois_dir, exist_ok=True)

        whois_cols = ["whois_org", "whois_net"]
        
        for dev in self.device_label:
            src_path = os.path.join(self.save_path, f"ipraw_{dev}.csv")
            if not os.path.exists(src_path):
                logging.warning("Cluster CSV not found for device %s: %s", dev, src_path)
                continue

            logging.info("Preparing whois features for device %s", dev)

            df = pd.read_csv(src_path)

            # 如果已经存在 whois_cols 中的列，则跳过该设备
            if all(col in df.columns for col in whois_cols):
                logging.info("[-] Whois columns already exist for device %s, skip.", dev)
                continue

            required_cols = ["org_name", "org_handle", "net_name", "net_handle"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logging.warning("[-] Device %s missing org/net columns: %s", dev, missing)
                continue

            df["whois_org"] = (
                df["org_name"].fillna("").astype(str)
                + "|"
                + df["org_handle"].fillna("").astype(str)
            )
            df["whois_net"] = (
                df["net_name"].fillna("").astype(str)
                + "|"
                + df["net_handle"].fillna("").astype(str)
            )

            df.to_csv(src_path, index=False)

    def parse_log_successful_clusters(self):
        """
        Scan all (device_type, perspective_name) combos by iterating over
        self.device_labels and self.perspective_names, and read the cluster IDs
        already recorded in each {dev}_cluster_summaries.json file.
        
        Returns:
            dict: {(device_type, perspective_name): set(successful_cluster_ids)}
                  Keys include ALL (dev, perspective) combos.
                  Values may be empty sets if no summaries file exists.
        """
        successful = {}  # {(device_type, perspective_name): set(successful_cluster_ids)}

        for dev in self.device_labels:
            for perspective_name in self.perspective_names:
                key = (dev, perspective_name)
                summaries_file = os.path.join(self.save_path, f"embedding_{perspective_name}", f"{dev}_cluster_summaries.json")
                if os.path.exists(summaries_file):
                    try:
                        with open(summaries_file, 'r', encoding='utf-8') as f:
                            summaries = json.load(f)
                        successful[key] = {int(s["cluster_id"]) for s in summaries}
                    except Exception as e:
                        logging.warning(f"Failed to read {summaries_file}: {e}")
                        successful[key] = set()
                else:
                    successful[key] = set()

        logging.info("Parsed successful clusters from summaries files:")
        for (dev, persp), ids in successful.items():
            logging.info(f"  {dev} - {persp}: {len(ids)} clusters succeeded")

        return successful

    def _save_sorted_report(self, report_file, summaries_file, device_type, cluster_summaries):
        """Save report and summaries files with entries sorted by cluster_id."""
        # Sort cluster_summaries by cluster_id
        cluster_summaries.sort(key=lambda x: x["cluster_id"])
        
        # Generate report text from sorted summaries
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"CLUSTER ANALYSIS REPORT - {device_type.upper()}")
        report_lines.append(f"Generated at: {pd.Timestamp.now()}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for s in cluster_summaries:
            analysis = s["analysis"]
            report_lines.append(f"\n{'=' * 80}")
            report_lines.append(f"CLUSTER {s['cluster_id']} - {s['device_count']} devices")
            report_lines.append(f"{'=' * 80}")
            report_lines.append(f"Common Patterns: {analysis.get('common_patterns', '')}")
            report_lines.append(f"Distribution Characteristics: {analysis.get('distribution_characteristics', '')}")
            report_lines.append(f"Notable Similarities: {analysis.get('notable_similarities', '')}")
            report_lines.append(f"Cluster Summary: {analysis.get('cluster_summary', '')}")
            report_lines.append("")
        
        # Add summary section
        report_lines.append("\n" + "=" * 80)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append(f"Total clusters analyzed: {len(cluster_summaries)}")
        report_lines.append(f"Total devices in analyzed clusters: {sum(c['device_count'] for c in cluster_summaries)}")
        report_lines.append("")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        with open(summaries_file, "w", encoding="utf-8") as f:
            json.dump(cluster_summaries, f, indent=4)

    def explain_report(self, device_type: str = "camera", llm: str = "CLAUDE", perspective_name: str = "whois", analyze_cols: list = [], recovery_cluster_ids: set = None):
        """
        Generate a report analyzing cluster distributions using Gemini LLM API.
        
        Args:
            device_type: Device type (e.g., "camera", "printer", "scada")
            llm: LLM service (OpenAI compatible)
            perspective_name: Perspective name (e.g., "whois", "os")
            analyze_cols: Columns to analyze
            recovery_cluster_ids: If set, only re-analyze these cluster IDs (error recovery mode)
        """
        
        # File paths
        # cluster_file = os.path.join(self.save_path, f"embedding_{perspective_name}", f"ipraw_{device_type}_embedding_{perspective_name}_cluster.csv")
        cluster_file = os.path.join(self.save_path, f"embedding_{perspective_name}", f"ipraw_{device_type}_embedding_{perspective_name}_pca.csv")
        ipraw_file = os.path.join(self.save_path, f"ipraw_{device_type}.csv")

        report_file = os.path.join(self.save_path, f"embedding_{perspective_name}", f"ipraw_{device_type}_cluster_report.txt")
        summaries_file = os.path.join(self.save_path, f"embedding_{perspective_name}", f"{device_type}_cluster_summaries.json")

        if not os.path.exists(cluster_file):
            logging.error(f"Cluster file not found: {cluster_file}")
            return
        
        if not os.path.exists(ipraw_file):
            logging.error(f"IP raw file not found: {ipraw_file}")
            return
        
        if recovery_cluster_ids is None:
            if os.path.exists(report_file) and os.path.exists(summaries_file):
                logging.info(f"Report file already exists: {report_file}")
                return
        else:
            logging.info(f"Error recovery mode: re-analyzing clusters {sorted(recovery_cluster_ids)} for {device_type} - {perspective_name}")
        
        # 加载各个角度聚类后的文件
        logging.info(f"Loading cluster data from {cluster_file}")
        cluster_df = pd.read_csv(cluster_file)
        
        # 加载原始IP数据
        logging.info(f"Loading IP raw data from {ipraw_file}")
        ipraw_df = pd.read_csv(ipraw_file)
        
        # Merge dataframes on IP
        merged_df = cluster_df[['ip', 'cluster']].merge(ipraw_df[['ip'] + analyze_cols], on='ip', how='left')
        
        # Group by cluster
        clusters = merged_df[merged_df['cluster'] != -1].groupby('cluster')
        
        logging.info(f"Found {len(clusters)} valid clusters (excluding noise cluster -1)")

        # In recovery mode, load existing summaries to merge results back
        existing_summaries = []
        if recovery_cluster_ids is not None and os.path.exists(summaries_file):
            with open(summaries_file, 'r', encoding='utf-8') as f:
                existing_summaries = json.load(f)
            # Remove entries for clusters that will be re-analyzed
            existing_summaries = [s for s in existing_summaries if s["cluster_id"] not in recovery_cluster_ids]
        
        # Analyze each cluster
        cluster_summaries = list(existing_summaries)

        total_analyzed_clusters = 0

        for cluster_id, group in clusters:
            if len(group) < 2:  # Skip clusters with only 1 device
                continue
            
            # In recovery mode, only process specified cluster IDs
            if recovery_cluster_ids is not None and cluster_id not in recovery_cluster_ids:
                continue
            
            logging.info(f"Analyzing cluster {cluster_id} with {len(group)} devices")
            
            # Prepare data for LLM analysis
            cluster_data = []
            for _, row in group.iterrows():
                cluster_data.append({
                    "ip": row['ip'],
                    **{col: row[col] if pd.notna(row[col]) else "" for col in analyze_cols}
                })

            perspective_info = self.perspective_info_config.get(perspective_name, None)
            if perspective_info is not None:
                perspective_prompt_info = perspective_info["prompt_info"]
            else:
                logging.error(f"Perspective {perspective_name} not found in perspective_info_config, skipping...")
                continue
            
            # Create prompt for LLM
            prompt = f"""Analyze the following cluster of {len(cluster_data)} IoT devices (cluster ID: {cluster_id}).
            
                        The devices have the following {perspective_prompt_info}:

                    """

            for i, device in enumerate(cluster_data[:10], 1):  # Limit to first 10 to avoid token limits
                prompt += f"{i}. IP: {device['ip']}\n"
                for col in analyze_cols:
                    prompt += f"   - {col}: {device[col]}\n"
                prompt += "\n"
            
            if len(cluster_data) > 10:
                prompt += f"... and {len(cluster_data) - 10} more devices\n\n"
            
            prompt += f"""
                        Please analyze this cluster and provide the following result in JSON format:
                        1. common_patterns: Common patterns in {perspective_prompt_info}
                        2. distribution_characteristics: Geographic or organizational distribution characteristics
                        3. notable_similarities: Any notable similarities among feature values or combinations of features that explain why these devices are clustered together.
                        4. cluster_summary: A brief summary (2-3 sentences) of what characterizes this cluster

                        Keep the analysis concise and focused on the most significant patterns.
                        Please strictly return the results in JSON format.
                        """
            
            try:
                # Call LLM API
                messages=[
                        {"role": "system", "content": f"You are an expert in data analysis and IoT device clustering. Analyze the IoT device fingerprint feature data obtained through active probing to identify patterns and similarities."},
                        {"role": "user", "content": prompt}
                    ]
                
                analysis_json = self.llm.chat_with_llm(llm, messages, whether_json=True)
                   
                success_status = analysis_json.get("common_patterns", "")
                if not success_status:
                    logging.error(f"{llm} Failed to generate result in correct format! {perspective_name} - {device_type} - {cluster_id}")
                    raise ValueError(f"LLM错误生成结果: {analysis_json}")
                
                logging.info(f"cluster {cluster_id} analysis result: {analysis_json}")

                cluster_summaries.append({
                    "cluster_id": cluster_id,
                    "device_count": len(group),
                    "analysis": analysis_json
                })
                
                logging.info(f"Completed analysis for cluster {cluster_id}")
                
                total_analyzed_clusters += 1

                # Save partial report (sorted by cluster_id)
                if total_analyzed_clusters % 10 == 0:
                    self._save_sorted_report(report_file, summaries_file, device_type, cluster_summaries)
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logging.error(f"Error analyzing cluster {cluster_id}: {e}")
        
        # Save all report (sorted by cluster_id)
        self._save_sorted_report(report_file, summaries_file, device_type, cluster_summaries)

        print(f"\nCluster analysis report generated successfully!")
        print(f"Report saved to: {report_file}")
        print(f"Analyzed {len(cluster_summaries)} clusters")

    def generate_explain_report(self, error_recovery: bool = False):
        if error_recovery:
            successful_map = self.parse_log_successful_clusters()
            if not successful_map:
                logging.info("No attempted (dev, perspective) combos found in log, nothing to recover.")
                return

            for (dev, perspective_name), successful_ids in successful_map.items():
                perspective_cols = self.perspective_info_config.get(perspective_name, {}).get("cols", None)
                if perspective_cols is None:
                    logging.warning(f"Perspective {perspective_name} not found in config, skipping recovery for {dev} - {perspective_name}")
                    continue

                # Load cluster CSV to determine all valid cluster IDs
                cluster_file = os.path.join(self.save_path, f"embedding_{perspective_name}", f"ipraw_{dev}_embedding_{perspective_name}_pca.csv")
                ipraw_file = os.path.join(self.save_path, f"ipraw_{dev}.csv")
                if not os.path.exists(cluster_file) or not os.path.exists(ipraw_file):
                    logging.warning(f"CSV files not found for {dev} - {perspective_name}, skipping recovery")
                    continue

                cluster_df = pd.read_csv(cluster_file)
                ipraw_df = pd.read_csv(ipraw_file)
                merged_df = cluster_df[['ip', 'cluster']].merge(ipraw_df[['ip'] + perspective_cols], on='ip', how='left')
                clusters = merged_df[merged_df['cluster'] != -1].groupby('cluster')
                all_valid_ids = {int(cid) for cid, grp in clusters if len(grp) >= 2}

                missing_ids = all_valid_ids - successful_ids
                if not missing_ids:
                    logging.info(f"All clusters already analyzed for {dev} - {perspective_name}, skipping.")
                    continue

                logging.info(f"Recovering {dev} - {perspective_name}: {len(missing_ids)} incomplete clusters {sorted(missing_ids)}")
                self.explain_report(device_type=dev,
                                    llm="CLAUDE",
                                    perspective_name=perspective_name,
                                    analyze_cols=perspective_cols,
                                    recovery_cluster_ids=missing_ids)
        else:
            for dev in self.device_labels:
                for perspective_name in self.perspective_names:
                    if perspective_name != "http":
                        continue
                    perspective_cols = self.perspective_info_config[perspective_name]["cols"]
                    self.explain_report(device_type=dev, 
                                        llm="CLAUDE",
                                        perspective_name=perspective_name, 
                                        analyze_cols=perspective_cols)
        
    def perspective_clustering(self, target: str = "os_embedding"):
        start_time = time.time()
        
        logging.info(f"Service analysis for {target} started at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        perspective_name = target.replace('_embedding', '')
        
        if perspective_name not in self.perspective_names:
            logging.error(f"{perspective_name} not supported!")
            return

        feature_cols = self.perspective_cluster_info[perspective_name]
        logging.info(f"Feature columns: {feature_cols}")

        whether_distribution_analysis = self.cluster_embedding(cols=feature_cols, perspective_name=perspective_name, whether_pca=True)
        # self.cluster_embedding(cols=feature_cols, perspective_name=perspective_name, whether_pca=False)
        logging.info(f"Service analysis for {target} finished at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not whether_distribution_analysis:
            self.dev_distribution(cols=feature_cols, perspective_name=perspective_name)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Embedding & Clustering for {target} finished in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # python cluster.py --target all --gpu 0 -- overall --report
    # python cluster.py --target all --gpu -1
    # python cluster.py --target dns --gpu 0
    # python cluster.py --overall --gpu 0
    # python cluster.py --hgt --gpu 0
    # python cluster.py --report --gpu -1
    # python cluster.py --report --recovery --gpu -1

    parser = argparse.ArgumentParser(description="Graph Clustering & Service Analysis")
    parser.add_argument("--target", type=str, default="", help="Target forperspective clustering, e.g. dns, os, sw")
    parser.add_argument("--gpu", type=int, default=1, choices=[-1, 0, 1], help="GPU device number to use (0 or 1), -1 for CPU only")

    parser.add_argument("--overall", action="store_true", help="Whether to perform overall embedding clustering")
    parser.add_argument("--hgt", action="store_true", help="Whether to cluster devices using HGT global embeddings")
    parser.add_argument("--report", action="store_true", help="Whether to geneate clustering report")
    parser.add_argument("--recovery", action="store_true", help="Whether to recover from errors in previous report generation")

    args = parser.parse_args()

    gc_ins = GraphClustering(gpu=args.gpu)

    if args.target:
        log_filename = f"cluster_{args.target}.log"
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logging.getLogger().addHandler(file_handler)

        if args.target == "all":
            for perspective_name in gc_ins.perspective_names:
                gc_ins.perspective_clustering(target=f"{perspective_name}_embedding")
        else:
            gc_ins.perspective_clustering(target=f"{args.target}_embedding")

        gc_ins.concat_perspective_embeddings()

    if args.overall:
        log_filename = f"cluster_overall.log"
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logging.getLogger().addHandler(file_handler)

        gc_ins.overall_cluster_embedding(contact_done=False)
    
    if args.hgt:
        log_filename = f"cluster_hgt.log"
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logging.getLogger().addHandler(file_handler)

        gc_ins.HGT_cluster_embedding()

    if args.report:
        log_filename = f"cluster_report.log"
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logging.getLogger().addHandler(file_handler)

        gc_ins.generate_explain_report(error_recovery=args.recovery)