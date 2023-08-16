import torch
import numpy as np


def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "faces", "mass", "evals", "evecs", "gradX", "gradY"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                if name in v.keys() and v[name] is not None:
                    v[name] = v[name].to(device)
            dict_shape[k] = v
        else:
            if type(v) is list:
                for ii,vv in enumerate(v):
                    dict_shape[k][ii] = vv.to(device)

            else:
                dict_shape[k] = v.to(device)

    return dict_shape
def collate(data):
    
    CAD, PC, Obj = {},{},{}

    for key in data[0][0].keys():
        if isinstance(data[0][0][key], np.ndarray):
            CAD.update({key : [torch.Tensor(d[0][key]) for d in data]})
            CAD[key] = torch.nn.utils.rnn.pad_sequence(CAD[key], batch_first=True)
        else:
            CAD[key] = None #torch.nn.utils.rnn.pad_sequence(CAD[key], batch_first=True)
            #CAD.update({key : [ torch.sparse_coo_tensor(idx, val, shape)   d[0][key] for d in data]})
        

    for key in data[0][2].keys():
        if isinstance(data[0][2][key], np.ndarray) and data[0][2][key].size>1:
            Obj.update({key : [torch.Tensor(d[2][key]) for d in data]})
            if key != 'P':
                Obj[key] = torch.nn.utils.rnn.pad_sequence(Obj[key], batch_first=True)
        else:
            Obj.update({key : [d[2][key] for d in data]})
            #Obj[key] = torch.Tensor(Obj[key])

    for key in data[0][1].keys():
        if isinstance(data[0][1][key], np.ndarray):
            PC.update({key : [torch.Tensor(d[1][key]) for d in data]})
            PC[key] = torch.nn.utils.rnn.pad_sequence(PC[key], batch_first=True)
        else:
            PC[key] = None 
    return CAD, PC, Obj

def collate_noprocess(data):
    for b in data:
        to_del= ["L", "gradX", "gradY"]
        #print(b)
        for v in to_del:
            b[0].pop(v)
            b[1].pop(v)

    return data
