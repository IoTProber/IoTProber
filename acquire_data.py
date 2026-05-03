import binascii
import re
import os
import sys
import time
from tkinter import N
import csv
import pandas as pd
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import logging
import requests
from requests.auth import HTTPBasicAuth
from util import *
from censys_platform import SDK, Port

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='a',
    filename=os.path.join(os.getcwd(), "acquire_data.log")
)


class CensysData:
    def __init__(self, censys_version, org_id=None, personal_access_token=None,uid=None, secret=None):
        self.base_path = os.getcwd()
        self.save_path = os.path.join(self.base_path, "rag_data")
        self.test_data_path = os.path.join(self.base_path, "platform_data/2026-04-06")
        self.fingerprints_path = os.path.join(self.base_path, "rag_data/fingerprint")

        self.censys_version = censys_version

        if self.censys_version == "search":
            if uid is None or secret is None:
                raise ValueError("uid and secret must be provided for censys search version")
            self.api_id = uid
            self.secret = secret
        
        if self.censys_version == "platform":
            self.org_id = org_id
            self.personal_access_token = personal_access_token   

            if self.org_id is None or self.personal_access_token is None:
                raise ValueError("personal_access_token and organization id must be provided for censys platform version")

            self.sdk = SDK(organization_id=self.org_id, personal_access_token=self.personal_access_token)
        
        self.censys_initialize()

        self.device_label_list = load_all_dev_labels()
        self.new_device_label_list = load_new_dev_labels()

        self.http_base_ports = [1883, 8883, 5678, 8355, 8356, 8884, 443, 1884]

        self.improve_category_non_tls = {
            "1": ["BRUTE FORCE", "Bad user name or password"],
            "2": ["RANDOM CLIENT ID", "Client identifier not valid"],
            "3": ["AUTHORIZATION PASS", "Not authorized"],
            "4": ["CHANGE PROTOCOL VERSION", "Unsupported protocol version"],
        }

        self.improve_category_tls = {
            "1": ["SET LOCAL ISSUER CA",
                  "raise error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)"],
            "2": ["ENHANCED TLS SS CA IN CHAIN",
                  "raise error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1007)"],
            "3": ["ENHANCED TLS SS CA",
                  "raise error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate (_ssl.c:1007)"],
            "4": ["UPDATE CA",
                  "raise error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1007)"],
            "5": ["CHANGE EE KEY",
                  "raise error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: EE certificate key too weak (_ssl.c:1007)"],
            "6": ["CHANGE CA",
                  "raise error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: invalid CA certificate (_ssl.c:1007)"],
            "7": ["CHANGE TLS VERSION", "raise error: [SSL: WRONG_VERSION_NUMBER] wrong version number (_ssl.c:1007)"],
            "8": ["UPDATE TLS VERSION",
                  "raise error: [SSL: TLSV1_ALERT_INSUFFICIENT_SECURITY] tlsv1 alert insufficient security (_ssl.c:1007)"]
        }

        self.check_fine_grained_status = ["Unspecified error", "Server unavailable",
                                          "exceed timeout", "refused by remote backend",
                                          "raise error: [WinError 10054] \u8fdc\u7a0b\u4e3b\u673a\u5f3a\u8feb\u5173\u95ed\u4e86\u4e00\u4e2a\u73b0\u6709\u7684\u8fde\u63a5\u3002"]

    def censys_initialize(self):
        
        if self.censys_version == "search":
            self.feature_cols = ["ip", "org_name", "org_handle", "net_name", "net_handle", "latitude", "longitude",
                            "reverse_dns", "operating_system_vendor", "operating_system_product", "operating_system_version",
                            "service_distribution", "service_software", "http_tags", "http_favicons", "http_favicons_hash", "service_banner", 
                            "service_certificate_hash", "certificate_leaf_subject", "certificate_issuer_chain", "certificate_subject_chain"]
        
            self.support_url = {
                "Host": {
                    "Search": "https://search.censys.io/api/v2/hosts/search",
                    "Aggregation": "https://search.censys.io/api/v2/hosts/aggregate",
                    "IP": "https://search.censys.io/api/v2/hosts/{ip}",
                    "Events": "https://search.censys.io/api/v2/experimental/hosts/{ip}/events",
                    "Names": "https://search.censys.io/api/v2/hosts/{ip}/names",
                    "Certificates": "https://search.censys.io/api/v2/hosts/{ip}/certificates",
                    "Comments": "https://search.censys.io/api/v2/hosts/{ip}/comments",
                }
            }
            self.location_query_fields = ["location.coordinates.latitude", "location.coordinates.longitude"]
            self.operating_system_query_fields = ["operating_system.product", "operating_system.vendor", "operating_system.version"]
            self.software_query_fields = ["services.software.vendor", "services.software.version"]
            self.basic_query_fields = ["ip", "services.service_name", "services.extended_service_name",
                                    "services.port", "services.banner_hex"]
            self.whois_query_fields = ["whois.network.name", "whois.network.handle",
                                    "whois.organization.name", "whois.organization.handle"]
            self.dns_query_fields = ["dns.names", "dns.reverse_dns.names"]
            self.tls_query_fields = ["services.tls.presented_chain.issuer_dn", "services.tls.presented_chain.subject_dn",
                                    # "services.tls.version_selected", "services.tls.cipher_selected"
                                    ]
            self.http_query_fields = ["services.certificate",
                                    "services.http.response.html_tags",
                                    "services.http.response.favicons.name",
                                    # "services.http.response.favicons.size",
                                    "services.http.response.favicons.shodan_hash"
                                    # "services.http.response.body_hashes"
                                    ]
            # 主要是证书信息
            self.certificate_query_fields = ["services.tls.certificates.leaf_data.subject_dn"]
            self.mqtt_query_fields = ["services.mqtt.connection_ack_return.return_value",
                                    "services.mqtt.subscription_ack_return.return_value"]

        elif self.censys_version == "platform":
            self.feature_cols = ["ip", "as-asn", "as-name", "as-bgp_prefix", "as-country_code", "as-info", 
                                "loc-latitude", "loc-longitude", "loc-continent", "loc-country", "loc-country_code", "loc-province", "loc-city", "loc-postal_code", "loc-timezone", "loc-info", 
                                "dns-reverse", 
                                "whois-network-handle", "whois-network-name", "whois-organization-handle", "whois-organization-name", "whois-info",
                                "os-vendor", "os-product", "os-version", "os-info", 
                                "service-distribution", 
                                "sw-vendors", "sw-products", "sw-versions", "sw-info", 
                                "hw-vendors", "hw-products", "hw-info",  
                                "cert-fingerprints", "cert-subjects", "cert-issuers", "cert-info", "tls-versions", 
                                "http-bodys", "http-tags", "http-favicon_urls", "http-favicon_hashes", "http-part-info", "http-info"]
            
            self.whois_query_fields = ["host.whois.network.name", "host.whois.network.handle",
                                    "host.whois.organization.name", "host.whois.organization.handle"]
            
            self.as_query_fields = ["host.autonomous_system.asn", "host.autonomous_system.organization", "host.autonomous_system.name", "host.autonomous_system.bgp_prefix", "host.autonomous_system.country_code"]
            
            self.location_query_fields = ["host.location.continent", "host.location.country", "host.location.province", "host.location.city",
            "host.location.postal_code", "host.location.country_code", "host.location.registered_country", "host.location.registered_country_code",
            "host.location.timezone", "host.location.coordinates.latitude", "host.location.coordinates.longitude"]
            
            self.operating_system_query_fields = ["host.operating_system.product", "host.operating_system.vendor", "host.operating_system.version"]
            self.software_query_fields = ["host.services.software.vendor", "host.services.software.product", 
            "host.services.software.version", "host.services.software.type"]

            self.hardware_query_fields = ["host.services.hardware.vendor", "host.services.hardware.product", 
            "host.services.hardware.version", "host.services.hardware.type"]

            self.basic_query_fields = ["host.ip", "host.services.protocol", "host.services.port", "host.services.banner_hex"]

            self.dns_query_fields = ["host.dns.names", "host.dns.reverse_dns.names"]

            self.tls_query_fields = ["host.services.tls.presented_chain.issuer_dn", "host.services.tls.presented_chain.subject_dn",
            "host.services.tls.version_selected", "host.services.tls.presented_chain.fingerprint_sha256"]

            self.http_query_fields = ["host.services.endpoints.http.body",
                                    "host.services.endpoints.http.html_tags",
                                    "host.services.endpoints.http.network_log.resources.url",
                                    "host.services.endpoints.http.favicons.name",
                                    "host.services.endpoints.http.favicons.hash_shodan"]
            # 主要是证书信息
            self.certificate_query_fields = ["host.services.cert.fingerprint_sha256",
                                            "host.services.cert.parsed.subject_dn", 
                                            "host.services.cert.parsed.issuer_dn"
                                            ]
        else:
            raise ValueError("Invalid censys version")

    def acquire_query_fields(self, protocol=None):

        if protocol == "http" or protocol == "https":

            field_length = (len(self.basic_query_fields) + len(self.location_query_fields) +
                            len(self.whois_query_fields) + len(self.operating_system_query_fields) +
                            len(self.http_query_fields) + len(self.certificate_query_fields) +
                            len(self.tls_query_fields) + len(self.software_query_fields) + len(self.hardware_query_fields) + 
                            len(self.dns_query_fields) + len(self.as_query_fields))

            print(f"Total censys fields: {field_length}")

            if self.censys_version == "search" and field_length > 25:
                raise Exception("Query http fields overflow")
            else:
                return (self.basic_query_fields + self.location_query_fields + self.operating_system_query_fields
                        + self.software_query_fields + self.hardware_query_fields + self.whois_query_fields + self.dns_query_fields +
                        self.tls_query_fields + self.http_query_fields + self.certificate_query_fields + self.as_query_fields)

        else:
            if self.censys_version == "search" and (len(self.basic_query_fields) + len(self.dns_query_fields) + len(self.whois_query_fields) +
                    len(self.dns_query_fields) + len(self.tls_query_fields) > 25):
                raise Exception("Judge query fields overflow")

            else:
                return self.basic_query_fields + self.whois_query_fields + self.dns_query_fields + self.tls_query_fields

    def acquire_censys_iot_devices(self, device_label: str, version: str, protocol: str = None, start_cursor=None, start_page_index=None):
        """
        根据特定IoT设备的标签，从censys获取设备的指纹信息
        """
        # result_dir = os.path.join(self.save_path, device_label)
        result_dir = os.path.join(self.test_data_path, device_label)

        # 如果已经提取了该设备的指纹信息，则无需重复提取
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        page_index = 0

        if self.censys_version == "search":
            if version == "v4":
                basic_query = {
                    "q": f"labels=iot and labels={device_label}",
                    "per_page": 100,
                    "virtual_hosts": "EXCLUDE",
                    "sort": "RELEVANCE",
                    "fields": self.acquire_query_fields(protocol)
                }
            else:
                basic_query = {
                    "q": f"(labels: ipv6) and labels=iot and labels={device_label}",
                    "per_page": 100,
                    "virtual_hosts": "EXCLUDE",
                    "sort": "RELEVANCE",
                    "fields": self.acquire_query_fields(protocol)
                }
        elif self.censys_version == "platform":
            if version == "v4":
                basic_query = {
                    "fields": self.acquire_query_fields(protocol),
                    "page_size": 100,
                    "query": f"(host.services.software.type: IOT or host.services.hardware.type: IOT) and host.services.labels.value = {device_label}"
            }
            else:
                basic_query = {
                    "fields": self.acquire_query_fields(protocol),
                    "page_size": 100,
                    "query": f"host.labels.value: IPV6 and (host.services.software.type: IOT or host.services.hardware.type: IOT) and host.services.labels.value = {device_label}"
            }
        else:
            print("Error Censys Version")
            return None

        first_query_success = False

        if start_cursor:
            page_index = start_page_index
            if self.censys_version == "search":
                basic_query["cursor"] = start_cursor
            else:
                basic_query["page_token"] = start_cursor

        fail_time = 0
        while not first_query_success:
            if self.censys_version == "search":
                response = requests.post(
                    self.support_url["Host"]["Search"],
                    auth=HTTPBasicAuth(self.api_id, self.secret),
                    headers={"Content-Type": "application/json", "accept": "application/json"},
                    data=json.dumps(basic_query)
                )

                if response.status_code == 200:
                    data = response.json()

                    page_index += 1
                    first_query_success = True

                    with open(os.path.join(result_dir, f"page{page_index}.json"), "w") as f:
                        json.dump(data, f, indent=4)

                    print(f"Device type: {device_label}- Page {page_index} data has been fetched!")
                    logging.info(f"Device type: {device_label}- Page {page_index} data has been fetched!")
                    time.sleep(1)

                else:
                    fail_time += 1
                    if fail_time > 1:
                        return
                    print(f"Error: {response.status_code} - {response.text}")
                    logging.error(f"Error: {response.status_code} - {response.text}")
                    time.sleep(10)
                    continue
            
            elif self.censys_version == "platform":

                response = self.sdk.global_data.search(search_query_input_body=basic_query)

                response_dict = response.model_dump()
                host_hit_result = response_dict["result"]

                if int(host_hit_result["result"]["total_hits"]) == 0:
                    print("No Device type: {device_label} record in Censys database!")
                    return None
                
                page_index += 1

                with open(os.path.join(result_dir, f"page{page_index}.json"), "w") as f:
                    json.dump(host_hit_result, f, indent=4)
                
                print(f"Device type: {device_label}- Page {page_index} data has been fetched!")
                
                first_query_success = True
            
        
        fail_time = 0
        if self.censys_version == "search":
            while data["result"]["links"]["next"] != "":
                basic_query["cursor"] = data["result"]["links"]["next"]

                response = requests.post(
                    self.support_url["Host"]["Search"],
                    auth=HTTPBasicAuth(self.api_id, self.secret),
                    headers={"Content-Type": "application/json", "accept": "application/json"},
                    data=json.dumps(basic_query)
                )

                if response.status_code == 200:
                    data = response.json()

                    page_index += 1

                    with open(os.path.join(result_dir, f"page{page_index}.json"), "w") as f:
                        json.dump(data, f, indent=4)

                    print(f"Device type: {device_label}- Page {page_index} data has been fetched!")
                    logging.info(f"Device type: {device_label}- Page {page_index} data has been fetched!")
                    time.sleep(1)

                else:
                    fail_time += 1
                    if fail_time > 1:
                        return
                    print(f"Error: {response.status_code} - {response.text}")
                    logging.error(f"Error: {response.status_code} - {response.text}")
                    time.sleep(10)
                    continue

        elif self.censys_version == "platform":
            while host_hit_result["result"]["next_page_token"] != "":
                
                if page_index > 49:
                    break

                basic_query["page_token"] = host_hit_result["result"]["next_page_token"]
                response = self.sdk.global_data.search(search_query_input_body=basic_query)

                response_dict = response.model_dump()
                host_hit_result = response_dict["result"]

                if int(host_hit_result["result"]["total_hits"]) != 0:
                    page_index += 1

                with open(os.path.join(result_dir, f"page{page_index}.json"), "w") as f:
                    json.dump(host_hit_result, f, indent=4)

                print(f"Device type: {device_label}- Page {page_index} data has been fetched!")
                logging.info(f"Device type: {device_label}- Page {page_index} data has been fetched!")
        
            logging.info(f"Device type: {device_label} All data has been fetched!")

    def acquire_iot_basic_info(self, version, protocol):
        """
        获取对应设备标签的指纹数据
        """
        for device_label in self.device_label_list:
            result_dir = os.path.join(self.test_data_path, device_label)
            start_cursor = None
            start_page_index = None

            if os.path.exists(result_dir):
                # 找到所有已有的 page{index}.json 文件
                existing_pages = {}
                for fname in os.listdir(result_dir):
                    match = re.match(r'^page(\d+)\.json$', fname)
                    if match:
                        existing_pages[int(match.group(1))] = fname

                if existing_pages:
                    last_index = max(existing_pages.keys())
                    last_page_file = os.path.join(result_dir, existing_pages[last_index])
                    try:
                        with open(last_page_file, "r", encoding="utf-8") as f:
                            last_page_data = json.load(f)

                        if self.censys_version == "search":
                            next_token = last_page_data.get("result", {}).get("links", {}).get("next", "")
                        else:
                            next_token = last_page_data.get("result", {}).get("next_page_token", "")

                        if next_token:
                            start_cursor = next_token
                            start_page_index = last_index
                            print(f"[Resume] {device_label}: resuming from page {last_index}, cursor={next_token[:30]}...")
                            logging.info(f"[Resume] {device_label}: resuming from page {last_index}")
                        else:
                            print(f"[Skip] {device_label}: all pages already fetched (no next token in page {last_index})")
                            logging.info(f"[Skip] {device_label}: all pages already fetched")
                            continue
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"[Warn] {device_label}: failed to read last page file {last_page_file}, starting fresh: {e}")
                        logging.warning(f"{device_label}: failed to read last page file: {e}")

            self.acquire_censys_iot_devices(device_label, version, protocol,
                                            start_cursor=start_cursor,
                                            start_page_index=start_page_index)

    def acquire_new_device_types(self, version: str, protocol: str = None, max_pages: int = 500):
        """
        获取MEDIA_SERVER和VPN新设备类型的指纹信息，要求同时满足是IoT设备
        每个设备类型最多收集max_pages页数据，每页保存为独立JSON文件到对应设备类型目录
        """
        new_device_types = ["MEDIA_SERVER", "VPN"]

        for device_label in new_device_types:
            result_dir = os.path.join(self.test_data_path, device_label)

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            page_index = 0

            if self.censys_version == "search":
                if version == "v4":
                    basic_query = {
                        "q": f"labels=iot and labels={device_label}",
                        "per_page": 100,
                        "virtual_hosts": "EXCLUDE",
                        "sort": "RELEVANCE",
                        "fields": self.acquire_query_fields(protocol)
                    }
                else:
                    basic_query = {
                        "q": f"(labels: ipv6) and labels=iot and labels={device_label}",
                        "per_page": 100,
                        "virtual_hosts": "EXCLUDE",
                        "sort": "RELEVANCE",
                        "fields": self.acquire_query_fields(protocol)
                    }
            elif self.censys_version == "platform":
                if version == "v4":
                    basic_query = {
                        "fields": self.acquire_query_fields(protocol),
                        "page_size": 100,
                        "query": f"(host.services.software.type: IOT or host.services.hardware.type: IOT) and host.services.labels.value = {device_label}"
                    }
                else:
                    basic_query = {
                        "fields": self.acquire_query_fields(protocol),
                        "page_size": 100,
                        "query": f"host.labels.value: IPV6 and (host.services.software.type: IOT or host.services.hardware.type: IOT) and host.services.labels.value = {device_label}"
                    }
            else:
                print("Error Censys Version")
                return None

            first_query_success = False
            fail_time = 0

            while not first_query_success:
                if self.censys_version == "search":
                    response = requests.post(
                        self.support_url["Host"]["Search"],
                        auth=HTTPBasicAuth(self.api_id, self.secret),
                        headers={"Content-Type": "application/json", "accept": "application/json"},
                        data=json.dumps(basic_query)
                    )

                    if response.status_code == 200:
                        data = response.json()
                        page_index += 1
                        first_query_success = True

                        with open(os.path.join(result_dir, f"page{page_index}.json"), "w") as f:
                            json.dump(data, f, indent=4)

                        print(f"Device type: {device_label} - Page {page_index} data has been fetched!")
                        logging.info(f"Device type: {device_label} - Page {page_index} data has been fetched!")
                        time.sleep(1)
                    else:
                        fail_time += 1
                        if fail_time > 1:
                            break
                        print(f"Error: {response.status_code} - {response.text}")
                        logging.error(f"Error: {response.status_code} - {response.text}")
                        time.sleep(10)
                        continue

                elif self.censys_version == "platform":
                    response = self.sdk.global_data.search(search_query_input_body=basic_query)
                    response_dict = response.model_dump()
                    host_hit_result = response_dict["result"]

                    if int(host_hit_result["result"]["total_hits"]) == 0:
                        print(f"No Device type: {device_label} record in Censys database!")
                        break

                    page_index += 1

                    with open(os.path.join(result_dir, f"page{page_index}.json"), "w") as f:
                        json.dump(host_hit_result, f, indent=4)

                    print(f"Device type: {device_label} - Page {page_index} data has been fetched!")
                    logging.info(f"Device type: {device_label} - Page {page_index} data has been fetched!")
                    first_query_success = True

            if not first_query_success:
                continue

            fail_time = 0
            if self.censys_version == "search":
                while data["result"]["links"]["next"] != "" and page_index < max_pages:
                    basic_query["cursor"] = data["result"]["links"]["next"]

                    response = requests.post(
                        self.support_url["Host"]["Search"],
                        auth=HTTPBasicAuth(self.api_id, self.secret),
                        headers={"Content-Type": "application/json", "accept": "application/json"},
                        data=json.dumps(basic_query)
                    )

                    if response.status_code == 200:
                        data = response.json()
                        page_index += 1

                        with open(os.path.join(result_dir, f"page{page_index}.json"), "w") as f:
                            json.dump(data, f, indent=4)

                        print(f"Device type: {device_label} - Page {page_index} data has been fetched!")
                        logging.info(f"Device type: {device_label} - Page {page_index} data has been fetched!")
                        time.sleep(1)
                    else:
                        fail_time += 1
                        if fail_time > 1:
                            break
                        print(f"Error: {response.status_code} - {response.text}")
                        logging.error(f"Error: {response.status_code} - {response.text}")
                        time.sleep(10)
                        continue

            elif self.censys_version == "platform":
                while host_hit_result["result"]["next_page_token"] != "" and page_index < max_pages:
                    basic_query["page_token"] = host_hit_result["result"]["next_page_token"]
                    response = self.sdk.global_data.search(search_query_input_body=basic_query)

                    response_dict = response.model_dump()
                    host_hit_result = response_dict["result"]

                    if int(host_hit_result["result"]["total_hits"]) != 0:
                        page_index += 1

                    with open(os.path.join(result_dir, f"page{page_index}.json"), "w") as f:
                        json.dump(host_hit_result, f, indent=4)

                    print(f"Device type: {device_label} - Page {page_index} data has been fetched!")
                    logging.info(f"Device type: {device_label} - Page {page_index} data has been fetched!")

            logging.info(f"Device type: {device_label} All data has been fetched!")

    def integrate_device_fingerprint(self):
        """
        整合同一类型的所有指纹信息
        """
        for dev_label in self.device_label_list:
            device_files = list_files_in_folder(os.path.join(self.fingerprints_path, dev_label))

            device_all_fingerprints = {}
            device_all_whois = []
            device_all_http = []
            device_all_https = []
            device_all_tls = []

            logging.info(f"[Step 0] Integrating device {dev_label} fingerprints...")
            print(f"[Step 0] Integrating device {dev_label} fingerprints...")

            save_path = os.path.join(self.fingerprints_path, dev_label, "fingerprint.json")

            for fingerprint_file in device_files:

                if not os.path.basename(fingerprint_file).startswith("page"):
                    continue

                with open(fingerprint_file, "r", encoding="utf-8") as f:
                    fingerprint_page = json.load(f)

                if "result" not in fingerprint_page.keys():
                    continue

                for ip_fingerprint in fingerprint_page["result"]["hits"]:
                    device_ip = ip_fingerprint["ip"]
                    ip_fingerprint.pop("ip")
                    device_all_fingerprints[device_ip] = ip_fingerprint

                    if "whois" in ip_fingerprint.keys():
                        device_all_whois.append(ip_fingerprint["whois"])

                    for service in ip_fingerprint["services"]:

                        if "extended_service_name" not in service.keys():
                            continue

                        if service["extended_service_name"] == "HTTP":
                            if "http" in service.keys():
                                device_all_http.append(service["http"])

                        if service["extended_service_name"] == "HTTPS":
                            if "http" in service.keys():
                                device_all_https.append(service["http"])
                            if "tls" in service.keys():
                                device_all_tls.append(service["tls"])

            with open(os.path.join(self.fingerprints_path, dev_label, "whois_fingerprint.json"), "w") as f:
                json.dump({"whois": device_all_whois}, f, indent=4)

            with open(os.path.join(self.fingerprints_path, dev_label, "http_fingerprint.json"), "w") as f:
                json.dump({"http": device_all_http}, f, indent=4)

            with open(os.path.join(self.fingerprints_path, dev_label, "https_fingerprint.json"), "w") as f:
                json.dump({"https": device_all_https}, f, indent=4)

            with open(os.path.join(self.fingerprints_path, dev_label, "tls_fingerprint.json"), "w") as f:
                json.dump({"tls": device_all_tls}, f, indent=4)

            logging.info(f"[Step 0] Integrating device {dev_label} fingerprints complete!")
            print(f"[Step 0] Integrating device {dev_label} fingerprints complete!")

            with open(save_path, "w") as f:
                json.dump(device_all_fingerprints, f, indent=4)

    def extract_features_from_host(self, host_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        仅适配最新CENSYS Platform API获取的数据，从其中host_v1数据中提取特征
        使用缩写命名: as=autonomous_system, loc=location, os=operating_system, 
        svc=service, cert=certificate, dns=dns, sw=software, hw=hardware
        """
        features = {}
        
        if not host_data or 'resource' not in host_data:
            return features
        
        resource = host_data['resource']
        
        # IP Address
        features['ip'] = resource.get('ip', '')
        
        # Autonomous System (as-*)
        if 'autonomous_system' in resource:
            as_info = resource['autonomous_system']
            as_keys = ["asn", "name", "bgp_prefix", "country_code"]
            features_as = ""
            for key in as_keys:
                value = as_info.get(key, '')
                features[f"as-{key}"] = value
                features_as += f"as-{key}: {value}"
            
            features['as-info'] = features_as

        # Location (loc-*)
        if 'location' in resource:
            loc = resource['location']
            loc_info = ['continent', 'country', 'country_code', 'province', 'city', 'postal_code','timezone']
            features_loc = ""

            if 'coordinates' in loc:
                coords = loc['coordinates']
                features['loc-latitude'] = coords.get('latitude', 'Unknown')
                features['loc-longitude'] = coords.get('longitude', 'Unknown')
            
            for key in loc_info:
                value = loc.get(key, '')
                features[f"loc-{key}"] = value
                if key in ["continent", "country", "province", "city"]:
                    features_loc += f"loc-{key}: {value}"

            features_loc += f"loc-latitude: {features['loc-latitude']} loc-longitude: {features['loc-longitude']}"
            features['loc-info'] = features_loc
        
        # DNS (dns-*)
        if 'dns' in resource:
            dns = resource['dns']
            if 'reverse_dns' in dns and 'names' in dns['reverse_dns']:
                features['dns-reverse'] = '|'.join(dns['reverse_dns']['names'])
        
        # Whois (whois-*)
        if "whois" in resource:
            whois = resource['whois']
            features['whois-network-handle'] = 'Unknown'
            features['whois-network-name'] = 'Unknown'
            features['whois-organization-handle'] = 'Unknown'
            features['whois-organization-name'] = 'Unknown'
            
            if "network" in whois:
                network = whois['network']
                if network is not None:
                    features['whois-network-handle'] = network.get('handle', 'Unknown')
                    features['whois-network-name'] = network.get('name', 'Unknown')
            if "organization" in whois:
                organization = whois['organization']
                if organization is not None:
                    features['whois-organization-handle'] = organization.get('handle', 'Unknown')
                    features['whois-organization-name'] = organization.get('name', 'Unknown')
            
            features["whois-info"] = f"network-handle: {features['whois-network-handle']}|network-name: {features['whois-network-name']}|organization-handle: {features['whois-organization-handle']}|organization-name: {features['whois-organization-name']}"
        
        # Operating System (os-*)
        if 'operating_system' in resource:
            os_info = resource['operating_system']
            features['os-vendor'] = os_info.get('vendor', 'Unknown')
            features['os-product'] = os_info.get('product', 'Unknown')
            features['os-version'] = os_info.get('version', 'Unknown')
            features["os-info"] = "vendor: " + features["os-vendor"] + "|product: " + features["os-product"] + "|version: " + features["os-version"]
        
        # Services (svc-*)
        if 'services' in resource:
            services = resource['services']
            protocols_ports = []
            
            sw_vendors = []
            sw_products = []
            sw_versions = []
            sw_info = []

            hw_vendors = []
            hw_products = []
            hw_versions = []
            hw_info = []

            cert_fingerprints = [] #所有服务的证书指纹
            cert_subjects = [] # 所有服务的证书主题
            cert_issuers = [] # 所有服务的证书颁发者
            certification_info = []  # 所有服务的证书信息分布，包括各服务的证书主题+颁发者

            tls_versions = []

            http_body = []  # 所有服务各自的html body内容综合
            http_tags = []  # 所有服务各自的http tags综合
            http_favicon_hashes = []
            http_favicon_urls = []

            http_part_info = [] # HTTP Tags和Favicons的综合表示
            http_info = []  # HTTP服务的综合表示
            
            for svc in services:
                # Protocol and Port Combination
                pp_value = "" 
                protocol = svc.get("protocol", 'Unknown Protocol')
                port = svc.get("port", 'Unknown Port')
                if protocol == "UNKNOWN":
                    protocol = 'Unknown Protocol'
                if port == "UNKNOWN":
                    port = 'Unknown Port'

                pp_value = protocol
                
                pp_value += "-{}".format(port)
                
                protocols_ports.append(pp_value)
                
                # Software (sw-*) 软件信息统计，包括厂商，产品，版本号
                if 'software' in svc:
                    software_list = []
                    for sw in svc['software']:
                        vendor = sw.get('vendor', 'Unknown')
                        product = sw.get('product', 'Unknown')
                        version = sw.get('version', 'Unknown')
                        sw_vendors.append(vendor)
                        sw_products.append(product)
                        sw_versions.append(version)
                        software_list.append(f"vendor: {vendor}|product: {product}|version: {version}")
                    
                    all_software = ",".join(software_list)
                    sw_info.append(f"{pp_value}: [{all_software}]")
                
                # Hardware (hw-*) 硬件信息统计，包括厂商，产品
                if 'hardware' in svc:
                    hardware_list = []
                    for hw in svc['hardware']:
                        vendor = hw.get('vendor', 'Unknown')
                        product = hw.get('product', 'Unknown')
                        version = hw.get('version', 'Unknown')
                        hw_vendors.append(vendor)
                        hw_products.append(product)
                        hw_versions.append(version)
                        hardware_list.append(f"vendor: {vendor}|product: {product}|version:{version}")
                    
                    all_hardware = ",".join(hardware_list)
                    hw_info.append(f"{pp_value}: [{all_hardware}]")
                
                # Certificate (cert-*) 证书信息统计，包括指纹，主题，颁发者
                if 'cert' in svc:
                    cert = svc['cert']
                    subject_dn = "Unknown"
                    issuer_dn = "Unknown"
                    tls_version_selected = "Unknown"

                    if 'fingerprint_sha256' in cert:
                        cert_fingerprints.append(cert['fingerprint_sha256'])
                    if 'parsed' in cert:
                        parsed = cert['parsed']
                        subject_dn = parsed.get('subject_dn', 'Unknown')
                        issuer_dn = parsed.get('issuer_dn', 'Unknown')
                        if subject_dn:
                            cert_subjects.append(subject_dn)
                        if issuer_dn:
                            cert_issuers.append(issuer_dn)
                
                    # TLS (tls-*) TLS版本信息统计
                    if 'tls' in svc:
                        tls = svc['tls']
                        if 'version_selected' in tls:
                            tls_version_selected = tls['version_selected']
                        tls_versions.append(f"{pp_value}: {tls_version_selected}")

                    certification_info.append(f"{pp_value}: [Subject: {subject_dn}|Issuer: {issuer_dn}|TLS: {tls_version_selected}]")

                # HTTP (http-*)
                if 'endpoints' in svc:
                    http_single_tags = ""
                    http_single_body = ""
                    http_single_favicon_hash = []
                    http_single_favicons = []

                    for index, endpoint in enumerate(svc['endpoints']):
                        if 'http' in endpoint:
                            http = endpoint['http']
                            
                            # HTML Bodys
                            if 'body' in http:
                                http_single_body += f"{{endpoint{index}: {http['body']}}},"
                            
                            # HTML Tags
                            if 'html_tags' in http:
                                endpoint_html_tags = '\n'.join(http['html_tags'])
                                http_single_tags += f"{{endpoint{index}: {endpoint_html_tags}}},"

                            # Favicons
                            if 'favicons' in http:
                                for fav in http['favicons']:
                                    if 'hash_shodan' in fav:
                                        http_single_favicon_hash.append(str(fav['hash_shodan']))
                                    if 'name' in fav:
                                        http_single_favicons.append(fav['name'])
                    
                    if not http_single_body:
                        http_single_body = "{}"
                    if not http_single_tags:
                        http_single_tags = "{}"
                    
                    http_body.append(f"{pp_value}: {http_single_body}")
                    http_tags.append(f"{pp_value}: {http_single_tags}")
                    http_favicon_urls.append(f"{pp_value}: [{', '.join(http_single_favicons)}]")
                    http_favicon_hashes.append(f"{pp_value}: [{', '.join(http_single_favicon_hash)}]")
                    http_part_info.append(f"{pp_value}: (html tags: {http_single_tags}|favicons: [{', '.join(http_single_favicons)}])")
                    http_info.append(f"{pp_value}: (html body: {http_single_body}|html tags: {http_single_tags}|favicons: [{', '.join(http_single_favicons)}])")

            # 将列表转换为字符串
            features["service-distribution"] = ",".join(protocols_ports)
            features['sw-vendors'] = ','.join(filter(None, sw_vendors))
            features['sw-products'] = ','.join(filter(None, sw_products))
            features['sw-versions'] = ','.join(filter(None, sw_versions))
            features["sw-info"] = ','.join(sw_info)

            features['hw-vendors'] = ','.join(filter(None, hw_vendors))
            features['hw-products'] = ','.join(filter(None, hw_products))
            features['hw-versions'] = ','.join(filter(None, hw_versions))
            features["hw-info"] = ",".join(hw_info)

            features['cert-fingerprints'] = ','.join(cert_fingerprints)
            features['cert-subjects'] = '|'.join(cert_subjects)
            features['cert-issuers'] = '|'.join(cert_issuers)
            features['cert-info'] = ','.join(certification_info)
            
            features['tls-versions'] = ','.join(tls_versions)
            
            features["http-bodys"] = ",".join(http_body)
            features['http-tags'] = ",".join(http_tags)
            features['http-favicon-urls'] = ",".join(http_favicon_urls)
            features['http-favicon-hashes'] = ",".join(http_favicon_hashes)
            features["http-part-info"] = ",".join(http_part_info)
            features['http-info'] = ",".join(http_info)
        
        return features

    def convert_json_to_csv(self, device_label: str = None):
        """
        将目录下的JSON文件转换为CSV格式
        
        Args:
            device_label: 指定设备类型，如果为None则处理所有设备类型
        """
        # 创建CSV输出目录
        csv_dir = os.path.join(self.test_data_path, "csv")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
            # print(f"Created CSV directory: {csv_dir}")
            logging.info(f"Created CSV directory: {csv_dir}")
        
        # 确定要处理的设备类型列表
        if device_label:
            device_labels = [device_label]
        else:
            # 获取test_data下的所有设备类型目录
            device_labels = [d for d in os.listdir(self.test_data_path) 
                           if os.path.isdir(os.path.join(self.test_data_path, d)) and d != 'csv']
        
        for dev_label in device_labels:
            device_dir = os.path.join(self.test_data_path, dev_label)
            
            if not os.path.exists(device_dir):
                logging.warn(f"Warning: Device directory not found: {device_dir}")
                continue
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing device type: {dev_label}")
            logging.info(f"{'='*60}")

            # 收集所有主机数据
            all_features = []
            
            # 遍历该设备类型目录下的所有JSON文件
            json_files = sorted([f for f in os.listdir(device_dir) if f.endswith('.json')], 
                              key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
            
            for json_file in json_files:
                json_path = os.path.join(device_dir, json_file)
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 检查数据结构
                    if 'result' not in data or 'hits' not in data['result']:
                        logging.warn(f"Warning: Invalid JSON structure in {json_file}")
                        continue
                    
                    # 处理每个hit
                    for hit in data['result']['hits']:
                        if 'host_v1' in hit:
                            features = self.extract_features_from_host(hit['host_v1'])
                            if features:
                                all_features.append(features)
                    
                    logging.info(f"  Processed {json_file}: {len(data['result']['hits'])} hosts")
                    
                except Exception as e:
                    logging.error(f"Error processing {json_file}: {str(e)}")
                    continue
            
            # 保存为CSV
            if all_features:
                csv_filename = f"ipraw_{dev_label}.csv"
                csv_path = os.path.join(csv_dir, csv_filename)
                
                # 使用pandas保存，确保所有列都存在
                df = pd.DataFrame(all_features)
                df.to_csv(csv_path, index=False, encoding='utf-8-sig', escapechar="\\")
                
                logging.info(f"\n✓ Saved {len(all_features)} records to {csv_filename}")
                logging.info(f"  Columns: {len(df.columns)}")
                logging.info(f"  Features: {', '.join(df.columns.tolist())}")
                logging.info(f"Converted {dev_label} to CSV: {len(all_features)} records")
            else:
                logging.warn(f"Warning: No data extracted for {dev_label}")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"CSV conversion completed!")
        logging.info(f"Output directory: {csv_dir}")
        logging.info(f"{'='*60}\n")
    
    def preprocess(self):
        """
        将所有设备类型的JSON文件转换为CSV格式
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"Starting preprocessing for all device types")
        logging.info(f"Starting preprocessing for all device types")
        logging.info(f"{'='*60}\n")
        
        for device_label in self.device_label_list:
            logging.info(f"Processing device type: {device_label}")
            logging.info(f"Processing device type: {device_label}")
            self.convert_json_to_csv(device_label)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Preprocessing completed for all device types!")
        logging.info(f"Preprocessing completed for all device types!")
        logging.info(f"{'='*60}\n")

    def filter_and_export_csv(self, device_label: str):
        """
        从获取的JSON文件中筛选设备IP：若某IP的指纹中任意服务的hardware type
        匹配rag_domain.json中任意设备类型或等于VPN，则过滤掉该IP。
        将剩余IP经extract_features_from_host转换为特征字典，
        保存到platform_data/csv/ipraw_{device_label}.csv。
        
        Args:
            device_label: 设备类型标签（如MEDIA_SERVER或VPN）
        """
        filter_types = set(t.upper() for t in self.device_label_list if t != device_label) | set(t.upper() for t in self.new_device_label_list if t != device_label)
        # filter_types =  set(t.upper() for t in self.new_device_label_list if t != device_label)
        
        print(f"filter_types: {filter_types}")
        device_dir = os.path.join(self.test_data_path, device_label)

        if not os.path.exists(device_dir):
            logging.error(f"Device directory not found: {device_dir}")
            print(f"Device directory not found: {device_dir}")
            return

        csv_dir = os.path.join(self.test_data_path, "csv")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        all_features = []
        filtered_count = 0
        total_count = 0

        json_files = sorted(
            [f for f in os.listdir(device_dir) if f.endswith('.json')],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )
        print(f"Found {len(json_files)} JSON files in {device_dir}")
        for json_file in json_files:
            json_path = os.path.join(device_dir, json_file)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'result' not in data or 'hits' not in data['result']:
                    logging.warning(f"Invalid JSON structure in {json_file}")
                    continue
                
                # print(f"Processing {len(data['result']['hits'])} hits from {json_file}")

                for hit in data['result']['hits']:
                    if 'host_v1' not in hit:
                        continue

                    host_data = hit['host_v1']
                    resource = host_data.get('resource', {})
                    services = resource.get('services', [])
                    total_count += 1

                    should_filter = False
                    
                    for svc in services:
                        if 'hardware' in svc:
                            for hw in svc['hardware']:
                                hw_type = hw.get('type', '')
                                if isinstance(hw_type, list):
                                    if any(t.upper() in filter_types for t in hw_type if t):
                                        should_filter = True
                                        break
                                elif hw_type and hw_type.upper() in filter_types:
                                    should_filter = True
                                    break
                            if should_filter:
                                break
                        if 'software' in svc and not should_filter:
                            for sw in svc['software']:
                                sw_type = sw.get('type', '')
                                if isinstance(sw_type, list):
                                    if any(t.upper() in filter_types for t in sw_type if t):
                                        should_filter = True
                                        break
                                elif sw_type and sw_type.upper() in filter_types:
                                    should_filter = True
                                    break
                            if should_filter:
                                break

                    if should_filter:
                        filtered_count += 1
                        continue

                    features = self.extract_features_from_host(host_data)
                    if features:
                        all_features.append(features)

            except Exception as e:
                logging.error(f"Error processing {json_file}: {str(e)}")
                continue

        logging.info(f"Device type: {device_label} - Total: {total_count}, Filtered: {filtered_count}, Remaining: {len(all_features)}")
        print(f"Device type: {device_label} - Total: {total_count}, Filtered: {filtered_count}, Remaining: {len(all_features)}")

        if all_features:
            csv_filename = f"ipraw_{device_label}.csv"
            csv_path = os.path.join(csv_dir, csv_filename)

            df = pd.DataFrame(all_features)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig', escapechar="\\")

            logging.info(f"Saved {len(all_features)} records to {csv_filename}")
            print(f"Saved {len(all_features)} records to {csv_filename}")
        else:
            logging.warning(f"No data remaining after filtering for {device_label}")
            print(f"No data remaining after filtering for {device_label}")

    def drift_ip_collection(self):
        """
        从test_data目录下获取各个设备类型的所有IP列表，去重后保存到drift_data目录
        每个设备类型的IP保存成一个csv文件ip_{device_label}.csv，特征列名为target_ip
        """
        drift_data_path = os.path.join(self.base_path, "drift_data")
        
        if not os.path.exists(drift_data_path):
            os.makedirs(drift_data_path)
            print(f"Created drift_data directory: {drift_data_path}")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Starting IP collection for drift detection")
        logging.info(f"{'='*60}\n")
        
        for dev_label in self.device_label_list:
            device_dir = os.path.join(self.test_data_path, dev_label)
            
            if not os.path.exists(device_dir):
                logging.warn(f"Warning: Device directory not found: {device_dir}")
                continue
            
            logging.info(f"Processing device type: {dev_label}")
            
            unique_ips = set()
            
            json_files = [f for f in os.listdir(device_dir) if f.endswith('.json')]
            
            for json_file in sorted(json_files):
                json_path = os.path.join(device_dir, json_file)
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if 'result' not in data or 'hits' not in data['result']:
                        continue
                    
                    for hit in data['result']['hits']:
                        if 'host_v1' in hit and 'resource' in hit['host_v1']:
                            ip = hit['host_v1']['resource'].get('ip', '')
                            if ip:
                                unique_ips.add(ip)
                    
                except Exception as e:
                    logging.error(f"  Error processing {json_file}: {str(e)}")
                    logging.error(f"Error processing {json_file}: {str(e)}")
                    continue
            
            if unique_ips:
                csv_filename = f"ip_{dev_label}.csv"
                csv_path = os.path.join(drift_data_path, csv_filename)
                
                df = pd.DataFrame({'target_ip': sorted(list(unique_ips))})
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                logging.info(f"  ✓ Saved {len(unique_ips)} unique IPs to {csv_filename}")
                logging.info(f"Collected {len(unique_ips)} unique IPs for {dev_label}")
            else:
                logging.warn(f"  Warning: No IPs found for {dev_label}")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"IP collection completed!")
        logging.info(f"Output directory: {drift_data_path}")
        logging.info(f"{'='*60}\n")

def main():
    # Collect data
    # python acquire_data.py --collect
    # python acquire_data.py --collect --org_id YOUR_ORG_ID --token YOUR_TOKEN
    
    # # Convert all device types to CSV
    # python acquire_data.py --convert
    # python acquire_data.py --convert --org_id YOUR_ORG_ID --token YOUR_TOKEN
    
    # # Collect drift IPs
    # python acquire_data.py --drift --org_id YOUR_ORG_ID --token YOUR_TOKEN
    
    # # Combine multiple operations
    # python acquire_data.py --collect --convert --drift --org_id YOUR_ORG_ID --token YOUR_TOKEN

    # Acquire New devices
    # python acquire_data.py --collect_new --filter_new

    # python acquire_data.py --filter_old

    import argparse
    
    parser = argparse.ArgumentParser(description='IoT Device Data Collection and Processing')
    parser.add_argument('--collect', action='store_true', help='Collect data from Censys')
    parser.add_argument('--collect_new', action='store_true', help='Collect MEDIA_SERVER and VPN device data from Censys')
    parser.add_argument('--filter_new', action='store_true', help='Filter and export MEDIA_SERVER and VPN CSV files')
    parser.add_argument('--filter_old', action='store_true', help='Filter and export CSV files for all device types in rag_devices.json')
    parser.add_argument('--convert', action='store_true', help='Convert JSON files to CSV')
    parser.add_argument('--drift', action='store_true', help='Collect IPs for drift detection')
    parser.add_argument('--org_id', type=str, help='Censys organization ID')
    parser.add_argument('--token', type=str, help='Censys personal access token')
    
    args = parser.parse_args()
    
    if not args.org_id or not args.token:
        # LH Censys Token
        # args.org_id = "1a124af4-f289-43b6-8695-b1cd29be627b"
        # args.token = "censys_2jgji8RD_Pn5PmdEqkpTCsgthv7NGqxSG"

        # HX Censys Token
        args.org_id = ""
        args.token = ""   

        # print("Error: --org_id and --token are required")
        # return
    
    cs = CensysData(
        censys_version="platform",
        org_id=args.org_id,
        personal_access_token=args.token
    )
    print()
    
    if args.collect:
        print("Starting data collection...")
        cs.acquire_iot_basic_info("v4", "http")
    
    if args.collect_new:
        print("Starting MEDIA_SERVER and VPN data collection...")
        cs.acquire_new_device_types("v4", "http")
    
    if args.filter_new:
        print("Starting filter and CSV export for new device types...")
        for dev in ["MEDIA_SERVER", "VPN"]:
            cs.filter_and_export_csv(dev)
    
    if args.filter_old:
        print("Starting filter and CSV export for rag_devices.json device types...")
        for dev in cs.device_label_list:
            print(f"\nProcessing device type: {dev}")
            cs.filter_and_export_csv(dev)

    if args.convert:
        print("Starting CSV conversion...")
        cs.preprocess()
    
    if args.drift:
        print("Starting drift IP collection...")
        cs.drift_ip_collection()
    
if __name__ == "__main__":
    main()
