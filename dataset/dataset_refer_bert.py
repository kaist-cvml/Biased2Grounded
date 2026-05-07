import os
BASE_DIR = os.environ.get("BASE_DIR", "/mnt/image-net-full/namin/ris")
import ast
import sys
import json
from tkinter.constants import TRUE
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as T
import random
import clip
import argparse
import h5py
from refer.refer import REFER
import pandas as pd
from collections import defaultdict


def safe_eval(x):
    if pd.isna(x) or x == "" or x is None:
        return {}
    return ast.literal_eval(x)

class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 preprocessor=None,
                 split='train',
                 prompt_ensemble=False,
                 coco_instance_gt=False,
                 mask2former=False,
                 is_multilingual=False,
                 refined=False,
                 konglish=False,
                 refined_konglish=False):
        '''

        :param args: args
        :param image_transforms: get_transforms(args), Resize, ToTensor, Normalize, T.Compose(transforms)
        :param target_transforms: None
        :param split: 'train' or 'val'
        '''
        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.preprocessor = preprocessor
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
        self.max_tokens = 20
        self.Cat_dict = self.refer.Cats
        self.is_multilingual = is_multilingual
        self.konglish = konglish
        self.refined_konglish = refined_konglish

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.sentence_raws = []
        self.cat_names = []

        for r in ref_ids:
            ref = self.refer.Refs[r]

            text_embedding = []
            sentence_raw_for_ref = []
            cat_name = self.Cat_dict[ref['category_id']]
            self.cat_names.append(cat_name)

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']

                if prompt_ensemble:
                    text = [template.format(sentence_raw) for template in ReferDataset.templates]
                    text = clip.tokenize(text)
                else:

                    text = clip.tokenize(sentence_raw)
                text_embedding.append(text)
                sentence_raw_for_ref.append(sentence_raw)

            self.input_ids.append(text_embedding)
            self.sentence_raws.append(sentence_raw_for_ref)

        self.coco_instance_gt = coco_instance_gt
        if self.coco_instance_gt:
            path2img = os.path.join(BASE_DIR, 'mscoco_data/images/mscoco/images/train2014')
            path2ann = os.path.join(BASE_DIR, 'mscoco_data/annotations/instances_train2014.json')

            from pycocotools.coco import COCO
            self.coco = COCO(path2ann)

            self.coco_instance_cat_dict = self.coco.cats

        self.mask2former = mask2former

        if self.is_multilingual:
            folder_name = os.path.join(BASE_DIR, 'assets/multilingual_meta_data')
            if refined:
                file_name = os.path.join(folder_name, f"multilingual_{args.dataset}_{args.splitBy}_{self.split}_refined_dropped.csv")
                self.text_var = 'text_refined'
            elif self.konglish:
                file_name = os.path.join(folder_name, f"multilingual_{args.dataset}_{args.splitBy}_{self.split}_dropped_konglish.csv")
                self.text_var = 'text_konglish'
                self.text_var2 = 'konglish_variants'
            elif self.refined_konglish:
                file_name = os.path.join(folder_name, f"multilingual_{args.dataset}_{args.splitBy}_{self.split}_refined_dropped_konglish.csv")
                self.text_var = 'text_refined_konglish'
                self.text_var2 = 'konglish_variants'
                
            else:
                file_name = os.path.join(folder_name, f"multilingual_{args.dataset}_{args.splitBy}_{self.split}.csv")
                self.text_var = 'text'

            self.df = pd.read_csv(file_name)
            if self.konglish or self.refined_konglish:
                self.df[f"{self.text_var2}"] = self.df[f"{self.text_var2}"].apply(safe_eval)
                n = len(self.df)
                assert n % 2 == 0

                self.df['group_id'] = np.repeat(
                    np.arange(1, n // 2 + 1),
                    2
                )
            else:
                n = len(self.df)
                assert n % 10 == 0

                self.df['group_id'] = np.repeat(
                    np.arange(1, n // 10 + 1),
                    10
                )


    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]
        this_cat_name = self.cat_names[index]

        img_dir = os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])

        try:
            img = Image.open(img_dir).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: File not found {img_dir}. Skipping to next sample.")
            return self.__getitem__((index + 1) % len(self.ref_ids))

        if self.preprocessor is not None:
            tensor_img = self.preprocessor(img)
        else:
            tensor_img = torch.tensor(np.array(img))
        
        sam_img = np.array(img)
        
        ref = self.refer.loadRefs(this_ref_id)
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])

        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        sentence_raw = self.sentence_raws[index]

        if self.konglish or self.refined_konglish:
            lang_dic = defaultdict(list)
            gpt_txt_dic = defaultdict(list)
            for sent in sentence_raw:
                temp_df = self.df[self.df[self.text_var] == sent]
                if not temp_df.empty:
                    expr_id = temp_df['group_id'].values[0]
                    temp_df2 = self.df[self.df['group_id'] == expr_id]
                    temp_df2 = temp_df2.reset_index()
                    assert temp_df2.shape[0] == 2
                    lang = temp_df2['language'][1]
                    txt = temp_df2[self.text_var][1]
                    lang_dic[lang].append(txt)
                    gpt_txts = temp_df2[self.text_var2][1]
                    gpt_txt_dic = gpt_txts['konglish_phrases'][0] # first konglish dic

        elif self.is_multilingual:
            lang_dic = defaultdict(list)
            for sent in sentence_raw:
                temp_df = self.df[self.df[self.text_var] == sent]
                if not temp_df.empty:
                    expr_id = temp_df['group_id'].values[0]
                    temp_df2 = self.df[self.df['group_id'] == expr_id]
                    temp_df2 = temp_df2.reset_index()
                    assert temp_df2.shape[0] == 10
                    for i in range(temp_df2.shape[0]):
                        lang = temp_df2['language'][i]
                        txt = temp_df2[self.text_var][i]
                        lang_dic[lang].append(txt)

        if self.coco_instance_gt:
            coco_instance_target = self.coco.loadAnns(self.coco.getAnnIds(this_img_id))
            BoxAnns = []
            MaskAnns = []
            cat_names = []
            for t in coco_instance_target:
                BoxAnn = t['bbox']
                BoxAnns.append(BoxAnn)

                MaskAnn = self.coco.annToMask(t)
                MaskAnn = T.ToTensor()(MaskAnn)
                MaskAnn = T.Resize((np.asarray(img).shape[0], np.asarray(img).shape[1]),antialias=True)(MaskAnn)
                MaskAnns.append(MaskAnn)

                cat_id = t['category_id']
                cat_name = self.coco_instance_cat_dict[cat_id]['name']
                cat_names.append(cat_name)
            MaskAnns = torch.stack(MaskAnns, dim=1).squeeze(0).type(torch.bool) if len(MaskAnns) != 0 else []

        else:
            BoxAnns = []
            MaskAnns = []
            cat_names = []

        if self.mask2former:
            resized_img = np.asarray(img)
        else:
            # resized_img = T.Resize(800)(img)
            img = T.ToTensor()(img)
            img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        data = dict(image=img, tensor_img=tensor_img, sam_img=sam_img, height=np.asarray(img).shape[-2], width=np.asarray(img).shape[-1],
                         file_name=img_dir, cat_name=this_cat_name, img_id=this_img_id,
                         coco_instance_gt=MaskAnns, coco_instance_gt_box=BoxAnns, coco_instance_cat=cat_names)

        if self.is_multilingual:
            if self.konglish or self.refined_konglish:
                return data, np.asarray(annot), sentence_raw, gpt_txt_dic 
            else:
                return data, np.asarray(annot), sentence_raw, lang_dic
        else:
            return data, np.asarray(annot), sentence_raw



def get_parser():
    parser = argparse.ArgumentParser(description='Beta model')
    parser.add_argument('--clip_model', default='RN50', help='CLIP model name', choices=['RN50', 'RN101', 'RN50x4', 'RN50x64'])
    parser.add_argument('--visual_proj_path', default='./pretrain/', help='')
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--split', default='val', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument('--refer_data_root', default='./refer/data/', help='REFER dataset root directory')

    return parser




if __name__ == '__main__':
    args = get_parser()
    image_transforms = T.Compose([T.Resize(args.img_size, args.img_size),
                                  T.ToTensor(),
                                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ds = ReferDataset(args,
                      image_transforms=image_transforms,
                      target_transforms=None,
                      eval=True)
