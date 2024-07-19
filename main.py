from functions import test_time_adapt_eval
from model import *

from copy import deepcopy

from PIL import Image

from flags import *

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from model import *

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, AttentionGuidedAugmenter, build_dataset
from utils.tools import set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import vision_transformer as vits


def main():
    
    set_random_seed(0)
    
    set_random_seed(SEED)
    print("Use GPU: {} for training".format(GPU))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    
    classnames = imagenet_classes
        
    model = OurCLIP(classnames=classnames, n_ctx=N_CTX, ctx_init=CTX_INIT)
    
    if WEIGHTS is not None:
        print("Use pre-trained soft prompt (CoOp) as initialization")
        pretrained_ctx = torch.load(WEIGHTS)['state_dict']['ctx']
        assert pretrained_ctx.size()[0] == N_CTX
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(pretrained_ctx)
            model.prompt_learner.ctx_init_state = pretrained_ctx
            
    model_state = None

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(ARCH))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert GPU is not None
        torch.cuda.set_device(GPU)
        model = model.cuda(GPU)

    # define optimizer
    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, LR)
    optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    # iterating through eval datasets
    datasets = SET.split("/")
    results = {}
    for set_id in datasets:
        base_transform = transforms.Compose([
            transforms.Resize(RESOLUTION, interpolation=BICUBIC),
            transforms.CenterCrop(RESOLUTION)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        
        if DINO:
            arch='vit_small'
            patch_size=8 
            checkpoint_key='teacher'
            model_dino = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
            for p in model_dino.parameters():
                p.requires_grad = False
            model_dino.eval()
            device = "cuda"
            model_dino.to(device)

            #load the weights
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            if checkpoint_key in state_dict:
                state_dict = state_dict[checkpoint_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            model_dino.load_state_dict(state_dict, strict=False)
            
            data_transform = AttentionGuidedAugmenter(base_transform, preprocess, model_dino, n_views=VIEWS, 
                                        augmix=len(set_id)>1)
        else:
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=VIEWS, 
                                        augmix=len(set_id)>1)
        batchsize = 1

        print("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        classnames_all = imagenet_classes
        classnames = []
        if set_id in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
            classnames = [classnames_all[i] for i in label_mask]
        
        model.reset_classnames(classnames)
            

        val_dataset = build_dataset(set_id, data_transform, DATA, mode="test")
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=0, pin_memory=True)
            
        results[set_id] = test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(1, LR, 1))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")


if __name__ == '__main__':
    main()