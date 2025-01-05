import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import  MinMaxScaler

def get_data():


    data = np.load('./traffic_data/beijing.npz')['bs_record']
    observed_values=data
    observed_values = np.expand_dims(observed_values, axis=-1)
    return observed_values



class Physio_Dataset(Dataset):
    def __init__(self, observed_series, eval_length=None, use_index_list=None,seed=0):

        self.observed_values = observed_series
        self.eval_length = self.observed_values.shape[1]
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "timepoints": np.arange(self.eval_length),
            "idx": index,
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, batch_size=16):
    observed_series = np.array(get_data())

    all_indices = np.random.permutation(len(observed_series))
    num_train = int(0.8 * len(observed_series))
    num_valid = int(0.2 * len(observed_series))
    train_indices = all_indices[:num_train]
    valid_indices = all_indices[num_train:num_train + num_valid]
    test_indices = valid_indices

    myscaler = MinMaxScaler(feature_range=(0, 1))
    base_shape = observed_series.shape
    myscaler.fit(observed_series[train_indices].reshape(-1,1))
    observed_series = myscaler.transform(observed_series.reshape(-1,1)).reshape(base_shape)



    train_dataset = Physio_Dataset(
        observed_series=observed_series,
        use_index_list=train_indices,
        seed=seed
    )

    valid_dataset = Physio_Dataset(
        observed_series=observed_series,
        use_index_list=valid_indices,
        seed=seed
    )

    test_dataset = Physio_Dataset(
        observed_series=observed_series,
        use_index_list=test_indices,
        seed=seed
    )
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

    return train_loader, valid_loader, test_loader, myscaler

