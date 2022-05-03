from unittest import result
import torch
import torchvision
from torchvision.transforms import transforms
import numpy as np
from torch.utils.data import Dataset
import os
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from PIL import Image
import pickle
from matplotlib import pyplot as plt
from myutils import exif_transpose, letterbox, make_divisible


def preprocess_img(imgs):
    shape1 = []
    for i, im in enumerate(imgs):
        # PIL Image
        im = np.asarray(exif_transpose(im))
        if im.shape[0] < 5:  # image in CHW
            im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
        im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
        s = im.shape[:2]  # HWC
        g = (640 / max(s))  # gain
        shape1.append([y * g for y in s])
        imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
    
    shape1 = [make_divisible(x, 32) for x in np.array(shape1).max(0)]  # inf shape
    x = [letterbox(im, shape1, auto=False)[0] for im in imgs]  # pad
    x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
    x = torch.from_numpy(x) / 255  # uint8 to fp16/32
    return x

def my_collate(data):
    # imgids
    img_ids = [x['img_id'] for x in data]

    # labels
    labels = torch.stack([x['label'] for x in data], dim=0)

    # img
    if data[0]['img'] is not None:
        #imgs = torch.stack([x['img'] for x in data], dim=0)
        imgs = [x['img'] for x in data]
        imgs = preprocess_img(imgs)
    else:
        imgs = None

    # caption
    if data[0]['cap'] is not None:
        cap_embs = [x['cap'] for x in data]
    else:
        cap_embs = None

    # img features
    # if data[0]['img_feature'] is not None:
    #     img_features = []
    #     for feature_num in range(len(data[0]['img_feature'])):
    #         img_features.append(torch.stack([x['img_feature'][feature_num] for x in data], dim=0))
    #     if len(img_features) == 1:
    #         img_features = img_features[0]
    # else:
    #     img_features = None

    if data[0]['img_feature'] is not None:
        img_features = torch.stack([x['img_feature'] for x in data], dim=0)
    else:
        img_features = None

    return dict(caps=cap_embs, img_ids=img_ids, imgs=imgs, 
                labels=labels, img_features=img_features)

class MultiLabelDataset(Dataset):
    def __init__(self, img_root=None, label_root=None, cap_root=None, 
                 img_feature_root=None, img_feature_level=None, phase='train'):
        self.img_root = img_root
        self.label_root = label_root
        self.cap_root = cap_root
        self.phase = phase
        self.img_feature_root = img_feature_root
        self.img_feature_level = img_feature_level

        # self.img_feature = []
        # if img_feature_root is not None:
        #     if type(img_feature_root) is str:
        #         img_feature_root = [img_feature_root]
        #     for f in img_feature_root:
        #         assert os.path.isfile(f) and f.endswith('.npy')
        #         data = np.load(f)
        #         self.img_feature.append(data) # N, 1024

        if img_feature_root is not None:
            if img_feature_level is None:
                assert os.path.isfile(img_feature_root) and img_feature_root.endswith('.npy')
            else:
                img_feature_root = img_feature_root[:-4]+'_'+str(img_feature_level)+'_feature.npy'

            self.img_feature = np.load(img_feature_root) # N, 1024

        # load label
        self.df = pd.read_csv(os.path.join(label_root, str(phase)+'.csv'))
        self.img_ids = self.df["ImageID"]

        if phase == 'train':
            self.labels = self.load_label(self.df)
        else:
            self.labels = None
    
        # load cap
        if cap_root is not None:
            self.cap_embs = self.load_caption_feature(cap_root)
        else:
            self.cap_embs = None

        # image transforms
        if phase == 'train':
            self.transforms = transforms.Compose([
                                    #transforms.Resize((image_size, image_size)),
                                    #torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.IMAGENET),
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])
        else:
            self.transforms = transforms.Compose([
                                    #transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])


    def load_label(self, df):
        labels = df["Labels"].apply(lambda x: [int(x) for x in x.split()])
        mlb = MultiLabelBinarizer(classes=range(1, 20))
        return torch.FloatTensor(mlb.fit_transform(labels))
        
    
    def load_caption_feature(self, cap_root):
        #caption_emb_file = os.path.join(cap_root, phase+'_caption_emb.txt')

        with open(cap_root, 'rb') as text:
            cap_embs = pickle.load(text)

        tensor_matrix = []
        for data in cap_embs:
            now_sent = []
            for arr in data:
                now_sent.append(torch.FloatTensor(arr))
            tensor_matrix.append(now_sent)
        cap_embs = tensor_matrix

        return cap_embs
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        results = {}

        results['img_id'] = self.img_ids[index]

        results['label'] = self.labels[index] if self.labels is not None else torch.zeros(19,) -1 # -1 means no label given

        if self.img_feature_root is not None:
            results['img_feature'] = torch.FloatTensor(self.img_feature[index])
        else:
            results['img_feature'] = None

        if self.img_root is not None:
            img = Image.open(os.path.join(self.img_root, self.img_ids[index]))
            results['img'] = img#self.transforms(img)
        else:
            results['img'] = None
            #results['img_feature'] = None

        if self.cap_root is not None:
            results['cap'] = self.cap_embs[index]
        else:
            results['cap'] = None

        return results


if __name__ == '__main__':
    dataset = MultiLabelDataset('/media/administrator/1305D8BDB8D46DEE/5329/multi-label-classification-competition-22/COMP5329S1A2Dataset/data', 
                                '/media/administrator/1305D8BDB8D46DEE/5329/',
                                '/media/administrator/1305D8BDB8D46DEE/5329/cap_embedding/'
                                )
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=my_collate)
    for i, data in enumerate(trainloader):
        for k, v in data.items():
            if k != 'caps':
                if k != 'img_ids':
                    print(k, v.shape)
        print(data['labels'])
        img = data['imgs']
        plt.imshow(img[0].permute(1,2,0).cpu().numpy())
        plt.show()
        break