from data_provider.data_loader import PSMSegLoader, MSLSegLoader, RedTideSegLoader, PredictSegLoader
from torch.utils.data import DataLoader

data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'RedTide': RedTideSegLoader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
    elif flag == 'pred':
        shuffle_flag = False
        Data = PredictSegLoader
    else:
        shuffle_flag = True

    drop_last = False
    batch_size = args.batch_size
    
    data_set = Data(
        args = args,
        root_path=args.root_path,
        win_size=args.seq_len,
        flag=flag,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
