import time

import math

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
from flags import *

from model import *

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from data.imagnet_prompts import imagenet_classes
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

# fits a polynomial regression model to the input data.
def polynomial_regression(x, y, degree=2):
    """
    Parameters:
        x (array-like): The input data (independent variable).
        y (array-like): The target data (dependent variable).
        degree (int): The degree of the polynomial regression model.
    
    Returns:
        model (LinearRegression): The trained polynomial regression model.
        poly_features (PolynomialFeatures): The polynomial features transformer.
    """

    x = np.array(x).reshape(-1, 1) # reshape x for sklearn
    y = np.array(y)
    
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)
    
    model = LinearRegression() # fit the polynomial regression model
    model.fit(x_poly, y)
    
    return model, poly_features

# computes the derivative of a polynomial given its coefficients.
def polynomial_derivative(coefficients):
    """  
    Parameters: coefficients (array-like): The coefficients of the polynomial.
    Returns: derivative_coefficients (array-like): The coefficients of the derivative polynomial.
    """

    degree = len(coefficients) - 1
    derivative_coefficients = np.array([coefficients[i] * (degree - i) for i in range(degree)])
    return derivative_coefficients

# evaluates a polynomial at given points x.
def evaluate_polynomial(coefficients, x):
    """
    Parameters:
        coefficients (array-like): The coefficients of the polynomial.
        x (array-like): The points at which to evaluate the polynomial.
    
    Returns: y (array-like) - the values of the polynomial at the given points.
    """
    y = np.polyval(coefficients, x)
    return y


def find_max_threashold(sorted_entropies):

    degree = 7
    model, poly_features = polynomial_regression(range(len(sorted_entropies)), sorted_entropies, degree)

    x_new = np.linspace(0, len(sorted_entropies), 100)
    x_new = x_new.reshape(-1, 1)
    y_new = model.predict(poly_features.fit_transform(x_new))
    coefficients = np.polyfit(range(len(x_new)), y_new, degree)
    derivative_coefficients = polynomial_derivative(coefficients)

    # evaluate the derivative at the new data points
    y_derivative = evaluate_polynomial(derivative_coefficients, x_new.flatten())    

    stazionario = 0
    ascending = (y_derivative[1] - y_derivative[0]) > 0
    for i in range(len(y_derivative)-1):
        if ascending :
            if y_derivative[i+1] < y_derivative[i]:
                stazionario = int(x_new[i][0])
                break
        else :
            if y_derivative[i+1] > y_derivative[i]:
                stazionario = int(x_new[i][0])
                break
            
            
    return stazionario


def select_confident_samples_ours(logits):
    
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    i = 0

    '''
    for histogram in logits_list:
        row_to_save = [bathc_entropy_list[i]]
        row_to_save.append(histogram)
        i += 1 
    '''

    #compute the fn of the sorted_indices, find the derivativies and the minimun/maximum (first occurence)
    min_threshold = TRESHOLD
    sorted_entropy = sorted(batch_entropy.tolist())
    max_threshold = find_max_threashold(sorted_entropy)

    min_loss = math.inf
    n_chosen = TRESHOLD
    
    idx = torch.argsort(batch_entropy, descending=False)[:min_threshold]
    
    loss = avg_entropy(logits[idx])
    
    for i in range(min_threshold+1, max_threshold): #0.1 - 0.2
        n = i
        idx = torch.argsort(batch_entropy, descending=False)[:n]
        loss = avg_entropy(logits[idx])
        if loss < min_loss:
            min_loss = loss
            n_chosen = n
    
    #print('n: ', n_chosen)

    idx = torch.argsort(batch_entropy, descending=False)[:n_chosen]

    # return logits[idx], idx, loss, min_threashold, weighted_avg
    return logits[idx], idx, loss

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def test_time_tuning(model, inputs, optimizer, scaler):
    
    selected_idx = None
    for j in range(TTA_STEPS):
        with torch.cuda.amp.autocast():
            
            output = model(inputs) 

            if OUR_SELECTION:
                if selected_idx is not None:
                    output = output[selected_idx]
                    loss = avg_entropy(output)
                else:
                    output, selected_idx, loss = select_confident_samples_ours(output)
            
            else:
                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = select_confident_samples(output, SELECTION_P)

                loss = avg_entropy(output)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
        
    return


def test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        model.reset()

    mean=[0.48145466, 0.4578275, 0.40821073]
    std=[0.26862954, 0.26130258, 0.27577711]

    # Unnormalization function
    unnormalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    
    
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert GPU is not None
        
        # for k in range(len(images)):
        #     save_path = f'augmentations/{k}.png'
        #     image = torch.squeeze(images[k], dim=0)
        #     image = unnormalize(image)
        #     image = image.permute(1, 2, 0).cpu().numpy()
        #     image = np.squeeze(image)
        #     image = Image.fromarray((image * 255).astype(np.uint8))
        #     image.save(save_path)
        
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(GPU, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(GPU, non_blocking=True)
            image = images
        target = target.cuda(GPU, non_blocking=True)
        
        images = torch.cat(images, dim=0)

        # reset the tunable prompt to its initial state
        
        if TTA_STEPS > 0:
            with torch.no_grad():
                model.reset()
        optimizer.load_state_dict(optim_state)
        test_time_tuning(model, images, optimizer, scaler)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(image)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)

    progress.display_summary()

    return [top1.avg, top5.avg]