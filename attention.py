import torch
import torchvision.transforms as pth_transforms
from PIL import Image
import vision_transformer as vits

def get_attention_maps(img, model,arch='vit_small', patch_size=8, checkpoint_key='teacher'):
       
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # # Build the model
    # model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    # for p in model.parameters():
    #     p.requires_grad = False
    # model.eval()
    # model.to(device)
   

    # Load pretrained weights
    # url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    # state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    # #state_dict = torch.load(pretrained_weights, map_location="cpu")
    # if checkpoint_key in state_dict:
    #     state_dict = state_dict[checkpoint_key]
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    # model.load_state_dict(state_dict, strict=False)

    # Open and preprocess the image
    img = img.unsqueeze(0).to(device)

    # Make the image divisible by the patch size
    w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - img.shape[3] % patch_size
    img = img[:, :, :w, :h]

    # Get attention maps
    attentions = model.get_last_selfattention(img)

    nh = attentions.shape[1]  # number of heads
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w // patch_size, h // patch_size)
    attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    return attentions

# Example usage:
# image_path = "path_to_your_image.jpg"
# pretrained_weights = "path_to_pretrained_weights.pth"
# attention_maps = get_attention_maps(image_path, pretrained_weights)
# print(attention_maps)
