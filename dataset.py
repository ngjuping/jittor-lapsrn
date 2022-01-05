import jittor as jt
from jittor.dataset import Dataset
import h5py

class LapSRNDataset(Dataset):
    def __init__(self, file_path, batch_size=64):
        super().__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.label_x2 = hf.get("label_x2")
        self.label_x4 = hf.get("label_x4")

        self.set_attrs(total_len=self.data.shape[0], batch_size=batch_size, shuffle=True)

    def __getitem__(self, index ):
        return jt.Var(self.data[index,:,:,:]).float(), jt.Var(self.label_x2[index,:,:,:]).float(), jt.Var(self.label_x4[index,:,:,:]).float() 

    