import pandas as pd
import tensorflow as tf
import numpy as np

from params import values_to_keep


def dataframe_to_tf_dataset(dataframe: pd.DataFrame, target_name: str):
    dataframe = dataframe.copy()
    labels = dataframe.pop(target_name)
    ds = tf.data.Dataset.from_tensor_slices(
        (dict(dataframe), labels)
    )
    return ds


train_data = pd.read_csv("./train_data.csv",
                         sep=",",
                         skipinitialspace=True,
                         na_values="?")

# check for missing values
# nan_columns = train_data.isna().any()
# print(nan_columns)

# 删除id列
train_data = train_data.drop(columns=["id"])
train_data = train_data.drop(columns=["id.orig_p"])
feature_names = train_data.columns.tolist()
# print(feature_names)

train_data['id.resp_p'] = np.where(train_data['id.resp_p'].isin(values_to_keep),
                                   train_data['id.resp_p'], 0)

train_data['id.resp_p'] = train_data['id.resp_p'].apply(str)

# category_mapping = dict(enumerate(train_data["Attack_type"].astype("category").cat.categories)) # 输出数字和字符串的对应关系
# print(category_mapping) # {0: 'ARP_poisioning', 1: 'DDOS_Slowloris', 2: 'DOS_SYN_Hping', 3: 'MQTT_Publish',
# 4: 'Metasploit_Brute_Force_SSH', 5: 'NMAP_FIN_SCAN', 6: 'NMAP_OS_DETECTION', 7: 'NMAP_TCP_scan',
# 8: 'NMAP_UDP_SCAN', 9: 'NMAP_XMAS_TREE_SCAN', 10: 'Thing_Speak', 11: 'Wipro_bulb'}
train_data["Attack_type"] = train_data["Attack_type"].astype("category").cat.codes


# 获取测试集
test_data = train_data.sample(frac=0.1, random_state=1234)
train_data = train_data.drop(test_data.index)

train_ds = dataframe_to_tf_dataset(train_data, "Attack_type")
test_ds = dataframe_to_tf_dataset(test_data, "Attack_type")

# splitting data into batches
batch_size = 4096

train_ds = train_ds.shuffle(16384, seed=1234).batch(batch_size).prefetch(1)
test_ds = test_ds.batch(batch_size)

# all features sans the target
feature_names.remove("Attack_type")

# list of categorical features represented as string
cat_str_feature_names = list(train_data.select_dtypes(include=["object"]).columns)

# list of categorical features represented as int (note: none in this dataset)
cat_int_feature_names = list(train_data.select_dtypes(include=["int64"]).columns)

# dimension of the embedding layer for categorical variables
cat_embed_dims = {
    "id.resp_p": 16,
    "proto": 2,
    "service": 8,
}
