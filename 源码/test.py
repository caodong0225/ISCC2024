import numpy as np
import pandas as pd
import tensorflow as tf
import time

from params import values_to_keep
from tabnet_model import TabNetEncoder

model = tf.keras.models.load_model("../模型/results(2)", custom_objects={"TabNetEncoder": TabNetEncoder})

dict_map = {0: 'ARP_poisioning', 1: 'DDOS_Slowloris', 2: 'DOS_SYN_Hping', 3: 'MQTT_Publish',
            4: 'Metasploit_Brute_Force_SSH', 5: 'NMAP_FIN_SCAN', 6: 'NMAP_OS_DETECTION', 7: 'NMAP_TCP_scan',
            8: 'NMAP_UDP_SCAN', 9: 'NMAP_XMAS_TREE_SCAN', 10: 'Thing_Speak', 11: 'Wipro_bulb'}

val_data = pd.read_csv("./test_data.csv",
                       sep=",",
                       skipinitialspace=True,
                       na_values="?")
val_data['id.resp_p'] = np.where(val_data['id.resp_p'].isin(values_to_keep),
                                 val_data['id.resp_p'], 0)
val_data['id.resp_p'] = val_data['id.resp_p'].apply(str)
val_data = val_data.drop(columns=["id.orig_p"])
data_output = {
    "id": [],
    "Attack_type": []
}
for index, row in val_data[:5].iterrows():
    sample = dict(row)
    id_ = sample.pop("id")
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = model.predict(input_dict, verbose=0)
    predicted_class = np.argmax(predictions)
    data_output["id"].append(id_)
    data_output["Attack_type"].append(dict_map[predicted_class])
    if int(id_) % 100 == 0:
        print(id_, time.asctime())
# 创建 DataFrame
df = pd.DataFrame(data_output)
# 指定 CSV 文件路径
csv_file_path = "../提交结果/submission.csv"
# 将 DataFrame 写入 CSV 文件
df.to_csv(csv_file_path, index=False)
print("CSV 文件已生成并写入数据。")
