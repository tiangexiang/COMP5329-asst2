import sys
sys.path.insert(0, '../nn-compression')

from slender.coding import Codec
import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torch.nn as nn
from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
import matplotlib

from src_files.models.tresnet.tresnet import InplacABN_to_ABN
from dataset import MultiLabelDataset

matplotlib.use('TkAgg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch MS_COCO infer')
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--model-path', type=str, default='tresnet_xl_COCO_640_91_4.pth')
parser.add_argument('--pic-path', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model-name', type=str, default='tresnet_xl')
parser.add_argument('--image-size', type=int, default=640)
# parser.add_argument('--dataset-type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.75)
parser.add_argument('--top-k', type=float, default=20)
# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)


def main():
    print('quantization')

    # parsing args
    args = parser.parse_args()

    backend = "fbgemm"

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args, load_head=True).cuda()
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state['model'], strict=True)
    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.half().eval()
    torch.save(model.state_dict(), 'quant_test2.pth')
   # model = model.float().eval()


    # rule = []
    # for m, v in model.named_parameters():
    #     if m[-6:] == 'weight':
    #         rule.append((m, 'huffman', 0, 0, 4))
    # print(len(rule))

    # codec = Codec(rule=rule)
    # encoded_model = codec.encode(model = model)


    # torch.save(encoded_model.state_dict(), 'quant_test2.pth')
    
    #from torch.quantization import quantize_fx
    # qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}  # An empty key denotes the default applied to all modules
    # model_prepared = quantize_fx.prepare_fx(model, qconfig_dict)
    # model_quantized = quantize_fx.convert_fx(model_prepared)
    # torch.save(model_quantized.state_dict(), 'quant_test.pth')

    # qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
    # # Prepare
    # model_prepared = quantize_fx.prepare_fx(model, qconfig_dict)

    # model = nn.Sequential(torch.quantization.QuantStub(), 
    #               model, 
    #               torch.quantization.DeQuantStub())

    # """Prepare"""
    # model.qconfig = torch.quantization.get_default_qconfig(backend)
    # torch.quantization.prepare(model, inplace=True)

    # Calibrate - Use representative (validation) data.
    # with torch.inference_mode():
    #     for batch_id, data in enumerate(tqdm(loader)):
    #         img_ids = data['img_ids']
    #         pic_path = os.path.join('/media/sda5/USYD/5329/multi-label-classification-competition-22/COMP5329S1A2Dataset/data', img_ids[0])
    #         im = Image.open(pic_path)
    #         im_resize = im.resize((args.image_size, args.image_size))
    #         np_img = np.array(im_resize, dtype=np.uint8)
    #         tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    #         tensor_batch = torch.unsqueeze(tensor_img, 0).float()#.half() # float16 inference
    #         model(tensor_batch)
    #         if batch_id > 10:
    #             break

    # # quantize
    # #model_quantized = quantize_fx.convert_fx(model_prepared)
    # torch.quantization.convert(model, inplace=True)
    # torch.save(model.state_dict(), 'quant_test2.pth')

if __name__ == '__main__':
    main()
