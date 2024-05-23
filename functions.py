import random
import numpy as np
from enum import Enum
import torch
from pprint import pprint
import os
from os import listdir

ARCHITECTURE = "RN50"

DEVICE = "cuda:0"
GPU = 0

CLIP_RESOLUTION = 224
TTA_STEP = 1

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate_single(sample, y, model):
    
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(sample)
    y = y.cuda(GPU, non_blocking=True)
    acc1, acc5 = accuracy(output, y, topk=(1, 5))
    
    return acc1, acc5     

def validate(val_loader, model, criterion, args, output_mask=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if GPU is not None:
                images = images.cuda(GPU, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(GPU, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                if output_mask:
                    output = output[:, output_mask]
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if i % 200 == 0:
                progress.display(i)
        progress.display_summary()

    return top1.avg    
        
def calculate_entropy(tensor_top_prob, top=0.1):
    entropy = - (tensor_top_prob.softmax(-1) * tensor_top_prob.log_softmax(-1)).sum(-1)
    idx = torch.argsort(entropy, descending=False)[:int(entropy.size()[0] * top)]
    return tensor_top_prob[idx], idx

def test_time_tuning(model, inputs, optimizer, scaler):
    selected_idx = None
    for j in range(TTA_STEP):
        with torch.cuda.amp.autocast():
            output = model(inputs) 

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = calculate_entropy(output, 0.1)

            loss = avg_entropy(output)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

def test_time_adaptation(validation_loader, model, model_state, optimizer, optim_state, scaler):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(validation_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    model.eval()
    with torch.no_grad():
        model.reset()
        
    # end = time.time()

    for i, (images, y) in enumerate(validation_loader):
        # print("STEP: ", i)
        assert GPU is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(GPU, non_blocking=True)
            image = images[0]
        else:
            pprint("Error!!!")
        images = torch.cat(images, dim=0)
        y = y.cuda(GPU, non_blocking=True)
        
        # print("----AFTER TTA----")
        
        if TTA_STEP > 0:
            with torch.no_grad():
                model.reset()
        optimizer.load_state_dict(optim_state)
        test_time_tuning(model, images, optimizer, scaler)  # Calcolo loss + backpropagation
            
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(image)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, y, topk=(1, 5))

        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        progress.display(i)
    progress.display_summary()

    return [top1.avg, top5.avg]

# Their implementation
def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def report(avgtop1, avgtop5):

    results_dir = 'reports'
    test_folder_name = 'test0'
    i = 0
    while os.path.exists(os.path.join(results_dir, test_folder_name)):
        i += 1
        test_folder_name = f'test{i}'

    new_test_dir = os.path.join(results_dir, test_folder_name)
    os.makedirs(new_test_dir)

    results_path = os.path.join(new_test_dir, 'report.txt')

    results = f'Average top 1: {avgtop1} \n' + f'Average top 5: {avgtop5} \n'

    with open(results_path, 'w') as f:
        f.write(results)
