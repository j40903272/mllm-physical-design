from datasets import CongestionDataset, IRDropDataset, DRCDataset
import os

def load_dataset(task, ann_file, dataroot, return_dict=False):
    dataroot = os.path.join(dataroot, task)
    
    if task == "congestion":
        return CongestionDataset(
            ann_file = ann_file, 
            dataroot = dataroot, 
            return_dict=True
        )
    
    elif task == "IR_drop":
        return IRDropDataset(
            ann_file = ann_file, 
            dataroot = dataroot, 
            return_dict=True 
        )
    
    elif task == "DRC":
        return DRCDataset(
            ann_file = ann_file, 
            dataroot = dataroot, 
            return_dict=True 
        )


# dataset = load_dataset("IR_drop", "train_N28.csv", "/data2/NVIDIA/CircuitNet-N28/Dataset")
