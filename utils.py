"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os.path as osp
import time

import numpy as np
from PIL import Image

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform
from tllib.modules.entropy import entropy

sys.path.append('../../..')
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset

import datasets as datasets1


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + ['Digits']


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform

    if dataset_name == "OfficeHome":
        dataset = datasets1.__dict__[dataset_name]
        train_source_dataset = dataset(root, source[0], download=True,
                                                            transform=train_source_transform)
        # train_target_dataset = dataset(root, target[0], download=True,
        #                                                     transform=train_target_transform)
        # val_dataset = test_dataset = dataset(root, target[0], download=True, transform=val_transform)
        train_target_dataset = dataset(root, target[0], download=True, split='UT',
                                                            transform=train_target_transform)
        val_dataset = test_dataset = dataset(root, target[0], download=True, split='UT', transform=val_transform)
        class_names = datasets1.OfficeHome.CLASSES
        num_classes = len(class_names)


    elif dataset_name == "DomainNet":
        dataset = datasets1.__dict__[dataset_name]
        train_source_dataset = dataset(root, source[0], download=True,
                                       transform=train_source_transform)
        train_target_dataset = dataset(root, target[0], download=True,
                                       transform=train_target_transform)
        val_dataset  = dataset(root, target[0], download=True, transform=val_transform)
        test_dataset = dataset(root, target[0], split='test', download=True, transform=val_transform)
        class_names = datasets1.DomainNet.CLASSES
        num_classes = len(class_names)



    elif dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)
    elif dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, start_idx, **kwargs):
            # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
            return MultipleDomainsDataset([dataset(task=task, **kwargs) for task in tasks], tasks,
                                          domain_ids=list(range(start_idx, start_idx + len(tasks))))

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform,
                                              start_idx=0)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform,
                                              start_idx=len(source))
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform,
                                     start_idx=len(source))
        # val_dataset = concat_dataset(root=root, tasks=source, download=True, transform=val_transform,
        #                              start_idx=0)
        if dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform,
                                          start_idx=len(source))
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(dataset_name)
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names



def get_dataset_mask(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform

    if dataset_name == "OfficeHome":
        dataset = datasets1.__dict__[dataset_name]
        train_source_dataset_mask = dataset(root, source[0], download=True, mask=True,
                                                            transform=train_source_transform)
        # train_target_dataset = dataset(root, target[0], download=True,
        #                                                     transform=train_target_transform)
        # val_dataset = test_dataset = dataset(root, target[0], download=True, transform=val_transform)
        train_target_dataset_mask = dataset(root, target[0], download=True, split='UT',mask=True,
                                                            transform=train_target_transform)

    elif dataset_name == "DomainNet":
        dataset = datasets1.__dict__[dataset_name]
        train_source_dataset_mask = dataset(root, source[0], download=True,mask=True,
                                       transform=train_source_transform)
        train_target_dataset_mask = dataset(root, target[0], download=True,mask=True,
                                       transform=train_target_transform)

    return train_source_dataset_mask, train_target_dataset_mask


def validate(val_loader, model, args,num_classes, device,m) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        # num_classes = 10
        class_all = torch.zeros(num_classes).to(device)
        class_all_new= torch.zeros(num_classes).to(device)
        class_right = torch.zeros(num_classes).to(device)
        class_predict = torch.zeros(num_classes,num_classes).to(device)
        zhixindu = torch.zeros(num_classes).to(device)
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            xxx=F.softmax(output,dim=1)
            ############################################
            # yyy=xxx.clone()
            # a=xxx.clone()
            # for i in range(output.size()[0]):
            #     yyy[i]=1./2. *(xxx[i]-torch.max(xxx[i]))
            # # yyy=torch.exp(yyy)
            # print(yyy)
            ###############################################
            lll, label_predict = torch.max(xxx, 1)

            # print(target)
            # print(label_predict)
            # print(xxx)

            # for i in range(output.size()[0]):
            #     if target[i] != label_predict[i] :
            #         print(target[i].item(), label_predict[i].item(), lll[i].item(),xxx[i][target[i].item()].item() )


            ##############################
            for i in range(output.size()[0]):
                if lll[i]>=0.65:
                    class_all_new[target[i]]+=1
                class_all[target[i]]+=1
                if target[i] == label_predict[i] :
                    class_right[target[i]]+=1
                class_predict[target[i],label_predict[i]] +=1

            for i in range(output.size()[0]):
                # zhixindu[target[i]] += 1/2 *(lll[i]-second_largest(xxx[i]))
                if target[i] == label_predict[i]:
                    zhixindu[target[i]] += 1/2 * (lll[i]-second_largest(xxx[i]))
                else:
                    zhixindu[target[i]] += 1/2 * (xxx[i][target[i]]-lll[i])

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # print(class_right/class_all)
        class_confusion_matrix = (class_predict.t()/class_all).t()
        class_confusion_matrix = class_confusion_matrix * 1000
        class_confusion_matrix = torch.round(class_confusion_matrix)
        class_confusion_matrix = class_confusion_matrix/1000
        # print(class_confusion_matrix)
        print(torch.diag(class_confusion_matrix))
        mean_class_acc = torch.diag(class_confusion_matrix).mean()
        print(mean_class_acc)
        # print(class_all)
        zhixindu = zhixindu / class_all
        # print(zhixindu)
        print(zhixindu.mean())
        # print(class_all_new)

        k = 14
        values, indices = torch.topk(m, k)
        tatol = 0
        for i in indices:
            z = torch.diag(class_confusion_matrix)
            tatol += z[i]
        tatol /= k
        print(tatol)


        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    # return top1.avg
    return mean_class_acc.item()*100

def second_largest(tensor):
    # 将张量转换为一维数组
    flattened = tensor.flatten()
    # 找到最大值的索引
    max_index = torch.argmax(flattened)
    # 将最大值置为负无穷大
    flattened[max_index] = float('-inf')
    # 找到第二大值的索引
    second_max_index = torch.argmax(flattened)
    # 返回第二大值
    return flattened[second_max_index]

def validate_1(val_loader, model, args,num_classes, device) :
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        # num_classes = 10
        predictions=[]

        class_all = torch.zeros(num_classes).to(device)
        class_right = torch.zeros(num_classes).to(device)
        class_predict = torch.zeros(num_classes, num_classes).to(device)
        zhixindu = torch.zeros(num_classes).to(device)
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            xxx=F.softmax(output/2,dim=1)
            ############################################
            # yyy=xxx.clone()
            # a=xxx.clone()
            # for i in range(output.size()[0]):
            #     yyy[i]=1./2. *(xxx[i]-torch.max(xxx[i]))
            # # yyy=torch.exp(yyy)
            # print(yyy)
            ###############################################
            lll, label_predict = torch.max(xxx, 1)
            for i in range(output.size()[0]):
                # zhixindu[target[i]] =zhixindu[target[i]]+1/2*(lll[i]-second_largest(xxx[i]))
                if target[i] == label_predict[i]:
                    zhixindu[target[i]] += 1/2 * (lll[i]-second_largest(xxx[i]))
                else:
                    zhixindu[target[i]] += 1/2 * (xxx[i][target[i]]-lll[i])
            lll = lll.cpu()
            label_predict = label_predict.cpu().tolist()

            predictions=predictions+label_predict

            #
            for i in range(output.size()[0]):
                class_all[target[i]]+=1
                if target[i] == label_predict[i] :
                    class_right[target[i]]+=1
                class_predict[target[i],label_predict[i]] +=1

            # for i in range(output.size()[0]):
            #     if target[i] == label_predict[i]:
            #         zhixindu[target[i]] += 1/2 * (lll[i]-second_largest(xxx))
            #     else:
            #         zhixindu[target[i]] += 1/2 * (target[i]-lll[i])




            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # print(class_right/class_all)
        class_confusion_matrix = (class_predict.t()/class_all).t()
        class_confusion_matrix = class_confusion_matrix * 1000
        class_confusion_matrix = torch.round(class_confusion_matrix)
        class_confusion_matrix = class_confusion_matrix/1000
        # print(class_confusion_matrix)
        print(torch.diag(class_confusion_matrix))
        mean_class_acc = torch.diag(class_confusion_matrix).mean()
        print(mean_class_acc)
        zhixindu = zhixindu / class_all
        # print(zhixindu)
        print(zhixindu.mean())




        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    # return top1.avg
    return mean_class_acc.item()*100,predictions







def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


def empirical_risk_minimization(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        # batch_size, num_classes = y_s.shape
        # predictions = F.softmax(y_s / args.temperature, dim=1)  # batch_size x num_classes
        # entropy_weight = entropy(predictions).detach()
        # entropy_weight =  torch.exp(-entropy_weight)
        # # entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight))  # batch_size x 1
        # l=torch.sum(entropy_weight)/batch_size
        # print(l)


        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def empirical_risk_minimization2(train_source_iter,train_source_iter_1, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch/10*8):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    for i in range(args.iters_per_epoch/10*2):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)





def empirical_risk_minimization_1(train_source_iter, model, head,optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    head.train()
    # mdd_single.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        f_s = model(x_s)
        y_s = head(f_s)
        # margin_loss = -mdd_single(f_s,y_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        # mdd_single.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)