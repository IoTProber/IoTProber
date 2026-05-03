import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__)))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from api import ProtocolGraph
from util import *
from geopy.geocoders import Nominatim
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qwen3_embedding_06b")

embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path,
                                        model_kwargs={"device": "cpu"},
                                        encode_kwargs={"normalize_embeddings": True})

result1 = embedding_model.embed_query("HTTPS-80: 485454502f312e3120323030204f4b0d0a446174653a20203c52454441435445443e0d0a5365727665723a206e67696e780d0a436f6e74656e742d4c656e6774683a20313736370d0a436f6e74656e742d547970653a20746578742f68746d6c0d0a")
result2 = embedding_model.embed_query("HTTPS-82: 485454502f312e3120323030204f4b0d0a446174653a20203c52454441435445443e0d0a5365727665723a20556e6b6e6f776e2f302e302055506e502f312e30205669726174612d456d5765622f52365f315f300d0a582d506f77657265642d42793a205048502f352e342e370d0a436f6e74656e742d4c656e6774683a203734350d0a436f6e74656e742d547970653a20746578742f68746d6c0d0a5365742d436f6f6b69653a2063737266746f6b656e3d556f3851523877694a6c6639504132554d657133765562465533764e557352660d0a")
print(np.shape(result1))
print(np.shape(result2))
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, minkowski

# Ensure vectors are 2D for sklearn/scipy functions where needed
v1 = np.array(result1).reshape(1, -1)
v2 = np.array(result2).reshape(1, -1)

# 1. Cosine similarity
cos_sim = cosine_similarity(v1, v2)[0, 0]

# 2. Euclidean distance
euc_dist = euclidean(result1, result2)

# 3. Manhattan (L1) distance (cityblock in SciPy)
man_dist = cityblock(result1, result2)

# 4. Minkowski distance (p=3 as an example)
mink_dist = minkowski(result1, result2, 3)

print(f"Cosine similarity: {cos_sim}")
print(f"Euclidean distance: {euc_dist}")
print(f"Manhattan distance: {man_dist}")
print(f"Minkowski distance (p=3): {mink_dist}")
