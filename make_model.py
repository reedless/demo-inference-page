import base64
import io
import json
from pathlib import Path

import learn2learn as l2l
import numpy as np
import torch
from PIL import Image
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.transforms.functional import resize


def remove_transparency(im, bg_colour=(255, 255, 255)):
    # Only process if image has transparency 
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL 
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format

        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

def trim_whitespace(image_torch):    
    left = 0
    while left < image_torch.size(1) and len(image_torch[:,left].unique()) == 1:
        left += 1

    right = image_torch.size(1) - 1
    while right > 0 and len(image_torch[:,right].unique()) == 1:
        right -= 1

    top = 0
    while top < image_torch.size(0) and len(image_torch[top,:].unique()) == 1:
        top += 1

    bottom = image_torch.size(0) - 1
    while bottom > 0 and len(image_torch[bottom,:].unique()) == 1:
        bottom -= 1
        
    if top > bottom or left > right:
        return image_torch
        
    return image_torch[top:bottom+1,left:right+1]

class WrapperModel(torch.nn.Module):
      def __init__(self, device, test, hidden = 64, adaptation = 10, ways = 2):
            super().__init__()
            self.device = device
            self.hidden = hidden
            self.adaptation = adaptation
            self.ways = ways

            features = torch.nn.Sequential(l2l.nn.Lambda(lambda x: x.view(-1, 1, 256, 256)),
                            l2l.vision.models.ConvBase(hidden=hidden, channels=1, max_pool=False, layers=5),
                            l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])),
                            l2l.nn.Lambda(lambda x: x.view(-1, hidden)))
            features.to(device)
            features.load_state_dict(torch.load(f'adapted_weights/adapted_anil_features_{test}_2shots_20steps_64hidden.pth', map_location=device))
            
            head = torch.nn.Linear(hidden, ways)
            head = l2l.algorithms.MAML(head, lr=0.1)
            head.to(device)
            head.load_state_dict(torch.load(f'adapted_weights/adapted_anil_head_{test}_2shots_20steps_64hidden.pth', map_location=device))
                
            self.model = torch.nn.Sequential(features, head)


      def forward(self, input):
            # [256, 256, 4] to [4, 256, 256]
            input = input.permute(2, 0, 1)

            # [4, 256, 256] to [256, 256]
            image_torch_resize = torch.sum(input, dim=0) / 4

            im_tensor = 255 - image_torch_resize
            kernel_tensor = torch.ones((1, 1, 3, 3))
            im_tensor = im_tensor.unsqueeze(0).unsqueeze(0)
            torch_result = torch.clamp(torch.nn.functional.conv2d(im_tensor, kernel_tensor, padding=2, dilation=2), 0, 255)
            torch_result = 255 - torch_result

            return self.model(torch_result[0][0])

def check_models_work(device):
    with open('GVT999.txt') as file:
        one_sample = file.readlines()

    tests = ['Complex', 'Polygon', 'Clock', 'Memory']
    for test in tests:
        # load sample and remove transparency
        b64_str = json.loads(one_sample[0])[test.lower()]['attempts'][-1]['image']
        base64_decoded = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(base64_decoded))
        im = remove_transparency(image).convert('L')
        image_np = np.array(im)
        image_torch = torch.tensor(image_np).float()

        # crop and resize
        image_torch_crop = trim_whitespace(image_torch).unsqueeze(0)
        image_torch_resize = resize(image_torch_crop, (256, 256))

        # stack it to mimic 4 channel image
        image_torch_resize = torch.cat([image_torch_resize, image_torch_resize, image_torch_resize, image_torch_resize]).permute(1, 2, 0)

        model = WrapperModel(device, test)
        model.eval()

        print(model(image_torch_resize))


def build_all_models(device):
    tests = ['Complex', 'Polygon', 'Clock', 'Memory']
    for test in tests:
        model = WrapperModel(device, test)
        model.eval()
        
        dummy_input = torch.zeros((256, 256, 4))
        torch.onnx.export(model, dummy_input, 'onnx_model.onnx', verbose=True)


if __name__ == '__main__':
    device = torch.device('cpu')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="make", help="make to make models or check to check on one sample")
    args = parser.parse_args()

    if args.mode == "make":
        build_all_models(device)
    elif args.mode == "check":
        check_models_work(device)
