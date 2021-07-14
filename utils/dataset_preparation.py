import os
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter
import mindspore.datset.vision.c_transforms as vision

def convert2mindrecord(filename="123.mindrecord",data=None, label=None):
    """
    Convert npy form dataset to MindRecord Style
    Args:

    Return:
        create a mindrecord file in path './filename'

    """
    if data is None:
        print("=> No data found.Please input data")   
    if label is None:
        print("=> No label found.Please input label")
    if len(data)!=len(label):
        print("=> the number of labels cannot match the data")

    if os.path.exists(filename):
        os.remove(filename)
        os.remove(filename + ".db")
    writer = FileWriter(file_name=filename)
    cv_schema = {"data":{"type":"bytes"}, "label":{"type":"int32"}}
    writer.add_schema(cv_schema, "it is a cv dataset")
    writer.add_index(["label"])
    
    data_buffer = []
    for i in range(len(data)):
        sample = {}
        sample["data"] = data[i]
        sample["label"] = label[i]
        data.append(sample)
        if i%10 == 0:
            writer.write_raw_data(data)
            data = []
        if i%100 == 0:
            writer.commit()
            print("{:5d} images are added into MindRecord file".format(i))

    if data:
        writer.write_raw_data(data)
    writer.commit()
