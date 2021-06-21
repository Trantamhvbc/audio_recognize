import glob
import os
import  pandas as pd

def get_all_path_file(path):

    res = []
    list_dir = [path]
    for path in list_dir:
        if os.path.isdir(path):
            tmp = os.listdir(path)
            for i in tmp:
                list_dir.append(os.path.join(path,i))
        elif os.path.isfile(path):
            res.append(path)
    return res


def create_df_save_to_file_audio(path_local, labels):
    res = []
    for label  in labels:
        path = os.path.join(path_local,label)
        res += get_all_path_file(path)
    return res
#
labels = ["Hòa tấu 1","Song tấu","Đơn tấu"]
data = create_df_save_to_file_audio(path_local = """C:\\Users\\Admin\\Desktop\\APP\\thu_uyen\\TrainingData""" , labels = labels)
df = pd.DataFrame(data=data,columns=["path"])
df.to_csv("datas_2.csv",index=False)
