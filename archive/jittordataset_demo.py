from jittor.dataset import Dataset

class YourDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=20)

    def __getitem__(self, k):
        return k, k*k

dataset = YourDataset().set_attrs(batch_size=10, shuffle=True)
for x, y in dataset:
    print(x)
    print(y)