import os.path



import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def imbalance_process(dataset,dataname,max ,p ,num_logics):
    """
    :param dataset:
    :param dataname:
    :param max:  The max number of classes
    :param p:   imbalanced ratio
    :return:  imbalanced dataset

    """
    # 非均衡比
    ind = []
    labels = dataset.targets
    cls_num = (np.unique(labels)).size


    #Rank officehome in descending order based on instance counts, and sample.
    class_num = np.unique(labels, return_counts=True)[1]  #class number
    d = np.argsort(class_num)  #Sort Index Values
    c = np.zeros_like(class_num)  #imbalanced sampling order
    for i in range(len(c)):
        c[d[i]] = len(c) - i - 1  # c[d[i]] denotes arrangement from small to large

    if dataname =='Ar' or dataname =='Pr' or dataname =='Cl' or dataname =='Rw' :
        num_logics = c

    # elif dataname =='r' or dataname =='c' or dataname =='p' or dataname =='s' :
    #     num_logics = c

    # max_size = np.unique(labels,return_counts= True)[1][0]
    max_size = np.unique(labels, return_counts=True)[1][np.where(np.array(num_logics)==0)[0][0]]
    # max_size = np.unique(labels, return_counts=True)[1].max()
    max =  max_size if max_size<max else max
    for i in range(cls_num):
        index = np.where(np.array(labels) == i)[0]
        class_num = round(max * (p ** (num_logics[i] / (cls_num - 1.0))))
        ind1 = index[0:class_num]
        # ind1 = index[-class_num:]
        ind.extend(ind1)
    ind.sort()


############################################################
    dataset.targets = np.array(labels)[ind].tolist()
    print(np.unique(dataset.targets, return_counts=True))
    # dataset.targets = torch.tensor(dataset.targets)
    # Samples are in the form of a list, which can only be retrieved using ind after being converted to array
    dataset.samples = np.array(dataset.samples)[ind].tolist()
    ##Array can only store elements of the same type, and changing the samples form to [str, str] -->needs to be converted to [str, int]
    for line in dataset.samples:
        line[1] = int(line[1])
    if dataname == 'MNIST' or dataname == 'USPS' or dataname == 'SVHN'or dataname == 'SVHNRGB' or dataname == 'MNISTRGB':
        directory_name = "data/digits/{}/image_list".format(dataname)
        file_name = "mnist_{}_{}".format(dataname, str(p))
    elif dataname =='Ar' or dataname =='Pr' or dataname =='Cl' or dataname =='Rw':
        directory_name = "data/office-home/image_list"
        file_name = "{}_{}".format(dataname, str(p))
    # elif dataname =='c' or dataname =='p' or dataname =='r'or dataname =='s':
    #     directory_name = "data/domainnet/image_list"
    #     file_name = "{}_{}".format(dataname, str(p))

    imbalance_data_list = os.path.join(directory_name, file_name)

    if os.path.exists(imbalance_data_list):
        os.remove(imbalance_data_list)

    with open(imbalance_data_list, 'a') as f:
        for i in range(len(dataset.samples)):
            # _,a=dataset.samples[i][0].split(dataname+'/')
            # f.write(a+" "+dataset.samples[i][1]+'\n')
            a = dataset.samples[i][0]
            f.write('{} {}\n'.format(a, dataset.samples[i][1]))

    dataset.data_list_file = imbalance_data_list + '.txt'
    # print(dataset.data_list_file)
    image_directory_name = "image_list"
    image_imbalance_data_list = os.path.join(image_directory_name, file_name)
    dataset.image_list['train'] = image_imbalance_data_list + '.txt'

    return dataset


def Reweight(num_class_list):
    beta = 0.999999
    effective_num = 1.0 - np.power(beta, num_class_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(num_class_list)
    weight = torch.FloatTensor(per_cls_weights)
    print(weight)
    return weight

class CSCE(nn.Module):

    def __init__(self, para_dict=None):
        super(CSCE, self).__init__()
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        scheduler = cfg.LOSS.CSCE.SCHEDULER
        self.step_epoch = cfg.LOSS.CSCE.DRW_EPOCH

        if scheduler == "drw":
            self.betas = [0, 0.999999]
        elif scheduler == "default":
            self.betas = [0.999999, 0.999999]
        self.weight = None

    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def reset_epoch(self, epoch):
        idx = (epoch-1) // self.step_epoch
        beta = self.betas[idx]
        self.update_weight(beta)
        print(beta)

    def forward(self, feature, x, target, **kwargs):
        # print(self.weight)
        return F.cross_entropy(x, target, weight= self.weight)