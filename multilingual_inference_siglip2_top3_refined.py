import os
BASE_DIR = os.environ.get("BASE_DIR", "/mnt/image-net-full/namin/ris")

from pydoc import text
from turtle import hideturtle
import torch
import torchvision.transforms as T
import spacy
import numpy as np
import tqdm
import cv2
import pandas as pd
from transformers import AutoModel, AutoProcessor
from torch.cuda.amp import autocast
from PIL import Image
from transformers.image_utils import load_image
import os
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional import to_pil_image

from dataset.dataset_refer_bert import ReferDataset
from model.backbone_new import CLIPViTFM
from utils import default_argument_parser, Compute_IoU, extract_noun_phrase, gen_dir_mask, extract_dir_phrase, extract_rela_word, relation_boxes, extract_nouns, mask_to_bbox_mask, mask_iou_diversity

import gem
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

Height, Width = 384, 384


def main(args, Height, Width):
    assert args.eval_only, 'Only eval_only available!'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("now using cuda: ",torch.version.cuda)
    else :
        print("now using CPU")

    gem_model = gem.create_gem_model(
        model_name='ViT-B/16', pretrained='openai', device=device
    )
    preprocess = gem.get_gem_img_transform()
    dataset = ReferDataset(args,
                           image_transforms=None,
                           target_transforms=None,
                           preprocessor = preprocess,
                           split=args.split,
                           is_multilingual=True,
                           refined=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    
    # Load SigLIP2-base-patch16-384 instead of CLIP
    print("Loading SigLIP2-base-patch16-384 model...")
    Model = CLIPViTFM(model_name='siglip2-base-patch16-384').to(device)
    Model.eval()
    print("SigLIP2 model loaded successfully!")

    nlp = spacy.load('en_core_web_lg')
    
    cum_I, cum_U =0, 0
    m_IoU = []
    cum_I_new, cum_U_new =0, 0
    m_IoU_new = []

    cum_I_final, cum_U_final = 0, 0
    m_IoU_final = []
    cum_I_final_new, cum_U_final_new = 0, 0
    m_IoU_final_new = []

    cum_I_topk, cum_U_topk =0, 0
    m_IoU_topk = []
    cum_I_new_topk, cum_U_new_topk =0, 0
    m_IoU_new_topk = []

    cum_I_final_topk, cum_U_final_topk = 0, 0
    m_IoU_final_topk = []
    cum_I_final_new_topk, cum_U_final_new_topk= 0, 0
    m_IoU_final_new_topk = []

    div_score, div_score_final = [], []

    r = 0.5
    alpha = 0.6
    softmax0 = torch.nn.Softmax(0)
    softmax0 = softmax0.to(device)
    k1=3
    k2=6

    fusion_mode = args.fusion_mode
    print(f"fusion mode={fusion_mode}")
    sam = sam_model_registry['default'](checkpoint=os.path.join(BASE_DIR, "checkpoints/sam_vit_h_4b8939.pth"))
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam,
                                               points_per_side=64,
                                               pred_iou_thresh=0.86,
                                               stability_score_thresh=0.92,
                                               crop_n_layers=1,
                                               crop_n_points_downscale_factor=2,
                                               min_mask_region_area=100,)
    tbar = tqdm.tqdm(data_loader)

    ########################## load data ##########################

    results_per_instance = []

    # Create directory for saving masks
    mask_save_dir = f'./result_log/masks_multilingual_{args.dataset}_{args.split}'
    os.makedirs(mask_save_dir, exist_ok=True)

    for i, data in enumerate(tbar):
        # if i == 10:
        #     break
        
        image, target, sentence_raw, lang_dic = data
        if lang_dic:
            target = target.to(device)

            target_new = mask_to_bbox_mask(target)

            image_path = data[0]['file_name'][0]
            img = Image.open(image_path)
            
            original_img = image['image'].to(device)
            ####### generate masks #######
            sam_img = np.array((data[0]['sam_img'][0]))
            sam_masks = mask_generator.generate(sam_img)
            masks = [torch.tensor(m['segmentation']) for m in sam_masks]
            masks = torch.stack(masks).to(device)

            boxes = [m['bbox'] for m in sam_masks]
            boxes = torch.tensor(boxes).to(device)

            ####### preprocess global and local images #######
            pixel_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1).to(masks.device)
            
            imagesrc = data[0]['sam_img']
            blurred = cv2.GaussianBlur(imagesrc[0].numpy().copy(), (15, 15), 0)

            global_imgs = []
            local_imgs = []

            for pred_box, pred_mask in zip(boxes, masks):
                pred_mask, pred_box = pred_mask.type(torch.uint8), pred_box.type(torch.int)
        
                global_img = imagesrc[0].numpy()
                if type(pred_mask) != type(imagesrc[0].numpy()):
                    mask = pred_mask.cpu().numpy()
                sharp_region = cv2.bitwise_and(
                    global_img,
                    global_img,
                    mask=np.clip(mask, 0, 255).astype(np.uint8),
                )
                inv_mask = 1 - mask
                blurred_region = (blurred * inv_mask[:, :, None]).astype(np.uint8)
                global_img = cv2.add(sharp_region, blurred_region)

                global_img = T.ToTensor()(global_img)
                global_img = to_pil_image(global_img.clamp(0,1))

                global_img = Model.processor(images=[load_image(global_img)], return_tensors="pt").to(device)
                global_imgs.append(global_img.pixel_values[0])
                
                pil_img = pil_to_tensor(img).to(device) / 255.0
                masked_image = pil_img * pred_mask[None, ...]
                masked_image = to_pil_image(masked_image.clamp(0,1))
                masked_image = Model.processor(images=[load_image(masked_image)], return_tensors="pt").to(device)
                local_imgs.append(masked_image.pixel_values[0])

            global_imgs = torch.stack(global_imgs,dim=0).to(device)
            local_imgs = torch.stack(local_imgs, dim=0).to(device)

            ####### calculate hybrid features #######
            with autocast():
                hybrid_features = Model(local_imgs=local_imgs, global_imgs=global_imgs, pred_masks=masks, fusion_mode=fusion_mode, masking_block=11)

            for j, (lang, sentences) in enumerate(lang_dic.items()):
                for sentence in sentences:
                    sentence = sentence[0].lower()
                    doc = nlp(sentence)

                    sentence_for_spacy = []

                    for doccnt, token in enumerate(doc):
                        if token.text == ' ':
                            continue
                        sentence_for_spacy.append(token.text)

                    sentence_for_spacy = ' '.join(sentence_for_spacy)

                    dirflag = extract_dir_phrase(sentence_for_spacy, nlp, False)
                    visual_feature = hybrid_features
                    
                    # Use SigLIP2 tokenizer instead of CLIP
                    sentence_token = Model.tokenizer(sentence_for_spacy, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
                    noun_phrase, not_phrase_index, head_noun = extract_noun_phrase(sentence_for_spacy, nlp, need_index=True)
                    
                    noun_phrase_token = Model.tokenizer(noun_phrase, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
                    sentence_features = Model.model.get_text_features(**sentence_token)
                    noun_phrase_features = Model.model.get_text_features(**noun_phrase_token)

                    text_ensemble = r * sentence_features + (1-r) * noun_phrase_features
                    score_clip = Model.calculate_score(visual_feature, text_ensemble)
                    
                    noun_phrases, nouns = extract_nouns(sentence_for_spacy, nlp)
                    # SigLIP2 uses 768-dimensional embeddings instead of 512
                    other_noun_features = torch.zeros(1, Model.model.config.text_config.hidden_size).to(device)
                    cnt_other_nouns = 0
                    for other_noun in noun_phrases:
                        noun_token = Model.tokenizer('This is a photo of '+other_noun, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
                        other_noun_features += Model.model.get_text_features(**noun_token)
                        cnt_other_nouns += 1
                    if cnt_other_nouns != 0:
                        other_noun_features = other_noun_features / cnt_other_nouns
                        
                    score_clip_Neg = Model.calculate_score(visual_feature, other_noun_features)

                    max_index_hybrid = torch.argmax(score_clip)
                    _, topk_indices = torch.topk(score_clip, k=3, dim=0)
                    result_seg_hybrid = masks[topk_indices]

                    _, m_IoU, cum_I, cum_U = Compute_IoU(result_seg_hybrid[0], target, cum_I, cum_U, m_IoU)
                    _, m_IoU_new, cum_I_new, cum_U_new = Compute_IoU(result_seg_hybrid[0], target_new, cum_I_new, cum_U_new, m_IoU_new)
    
                    div_score =  mask_iou_diversity(result_seg_hybrid, div_score)
                    _, m_IoU_topk, cum_I_topk, cum_U_topk = Compute_IoU(result_seg_hybrid, target, cum_I_topk, cum_U_topk, m_IoU_topk, top_k=True)
                    _, m_IoU_new_topk, cum_I_new_topk, cum_U_new_topk = Compute_IoU(result_seg_hybrid, target_new, cum_I_new_topk, cum_U_new_topk, m_IoU_new_topk, top_k=True)

                    
                    score_clip = softmax0(score_clip)
                    score_clip_Neg = softmax0(score_clip_Neg)
                    
                    ######## Spatial Relationship Guidance ########
                    relaflag = extract_rela_word(sentence_for_spacy, nlp)
                    if k1 > len(score_clip):
                        k1 = len(score_clip)
                    if k2 > len(score_clip_Neg):
                        k2 = len(score_clip_Neg)
                    _, maxidxs = torch.topk(score_clip.view(-1),k=k1)
                    _, maxNegidxs = torch.topk(score_clip_Neg.view(-1),k=k2)

                    noun_phrases, nouns = extract_nouns(sentence_for_spacy, nlp)
                    topscores = np.zeros(k1)
                    if len(nouns)==0:
                        for idx_i in range(k1):                  #idx_i in 1,2,3,4,5
                            for idx_j in maxidxs:           
                                topscores[idx_i]=topscores[idx_i]+relation_boxes(boxes[maxidxs[idx_i]],boxes[idx_j],score_clip[maxidxs[idx_i]][0],score_clip[idx_j][0],relaflag)
                    else:
                        for idx_i in range(k1):                  #idx_i in 1,2,3,4,5
                            for idx_j in maxNegidxs:            #idx_j in boxes' idx
                                topscores[idx_i]=topscores[idx_i]+relation_boxes(boxes[maxidxs[idx_i]],boxes[idx_j],score_clip[maxidxs[idx_i]][0],score_clip_Neg[idx_j][0],relaflag)
                    
                    topscores = torch.Tensor(topscores).to(device)
                    topscores = softmax0(topscores)
                    
                    ########  Spatial Coherence Guidance ########
                    score_gem_list=[]
                    imgattn = gem_model(data[0]['tensor_img'].to(device), [noun_phrase])[0]
                    imgattn = T.Resize((int(data[0]['height']), int(data[0]['width'])), antialias=True)(imgattn)[0]
                    imgattn = imgattn.to(device)

                    imgattn = (imgattn-imgattn.min()) / (imgattn.max()-imgattn.min())

                    pmask = gen_dir_mask(dirflag, imgattn.shape[0], imgattn.shape[1], imgattn.device)
                    imgattn = imgattn * pmask    # Spatial Position Guidance

                    imgattn = imgattn / imgattn.mean()
                    
                    if relaflag == "big":
                        black = 1.95
                    elif relaflag == "small":
                        black = 1.5
                    else:
                        black = 1.8
                        
                    for pred_mask in masks:
                        pred_mask = pred_mask.type(torch.uint8)
                        score_gemtmp = (imgattn * (2-black) * pred_mask/(pred_mask.sum())).sum() - (imgattn * black * (1 - pred_mask) / ((1 - pred_mask).sum())).sum()
                        score_gem_list.append(torch.Tensor([score_gemtmp]))
                    score_gem = torch.stack(score_gem_list,dim=0)
                    score_gem=score_gem.to(device)
                    
                    for idx_i in range(k1):      
                        topscores[idx_i]=topscores[idx_i] * (1 - alpha) + alpha * score_gem[maxidxs[idx_i]][0]

                    max_index_final = torch.argmax(topscores)
                    _, topk_indices = torch.topk(topscores, k=3, dim=0)
                    max_index_finals = maxidxs[topk_indices]
                    result_seg_final = masks[max_index_finals]

                    _, m_IoU_final, cum_I_final, cum_U_final = Compute_IoU(result_seg_final[0], target, cum_I_final, cum_U_final, m_IoU_final)
                    _, m_IoU_final_new, cum_I_final_new, cum_U_final_new = Compute_IoU(result_seg_final[0], target_new, cum_I_final_new, cum_U_final_new, m_IoU_final_new)

                    div_score_final = mask_iou_diversity(result_seg_final, div_score_final)
                    _, m_IoU_final_topk, cum_I_final_topk, cum_U_final_topk = Compute_IoU(result_seg_final, target, cum_I_final_topk, cum_U_final_topk, m_IoU_final_topk, top_k=True)
                    _, m_IoU_final_new_topk, cum_I_final_new_topk, cum_U_final_new_topk = Compute_IoU(result_seg_final, target_new, cum_I_final_new_topk, cum_U_final_new_topk, m_IoU_final_new_topk, top_k=True)

                    # Save per-instance record
                    results_per_instance.append({
                        "index": i,
                        "image_path": image_path,
                        "language": lang,
                        "sentence_raw": sentence_raw,
                        "sentence": sentence_for_spacy,
                        "language": lang,
                        "mask_pred_hybrid": int(max_index_hybrid.cpu()),
                        "mask_pred_final": int(max_index_final.cpu()),

                        "IoU_hybrid": float(m_IoU[-1]),       # latest instance IoU
                        "IoU_final": float(m_IoU_final[-1]),  # latest instance IoU final
                        "IoU_hybrid_mask": float(m_IoU_new[-1]),       # latest instance IoU
                        "IoU_final_mask": float(m_IoU_final_new[-1]),  # latest instance IoU final

                        "IoU_hybrid_topk": float(m_IoU_topk[-1]),       # latest instance IoU
                        "IoU_final_topk": float(m_IoU_final_topk[-1]),  # latest instance IoU final
                        "IoU_hybrid_mask_topk": float(m_IoU_new_topk[-1]),       # latest instance IoU
                        "IoU_final_mask_topk": float(m_IoU_final_new_topk[-1]),  # latest instance IoU final
                        "diversity_score_final": float(div_score_final[-1])
                    })

    df = pd.DataFrame(results_per_instance)
    df.to_csv(f"./result_log/per_instance_multilingual_siglip2_{args.dataset}_{args.split}_refined_{i}.csv", index=False)
    print("Saved per-instance CSV:", f"./result_log/per_instance_multilingual_siglip2_{args.dataset}_{args.split}_refined_{i}.csv")

    f = open('./result_log/result_log_Multilingual_siglip2_refined.txt', 'a')
    f.write(f'\n\n SigLIP2-base-patch16-384 Model'
            f'\nDataset: Multilingual / {args.dataset} / {args.split}'
            f'\nOverall IoU / mean IoU')

    overall = cum_I * 100.0 / cum_U
    mean_IoU = torch.mean(torch.tensor(m_IoU)) * 100.0

    f.write(f'\npure hybridgl: {overall:.2f} / {mean_IoU:.2f}, data size: {i}')  
    overall_final = cum_I_final * 100.0 / cum_U_final
    mean_IoU_final = torch.mean(torch.tensor(m_IoU_final)) * 100.0

    f.write(f'\nhybridgl w/ spatial guidance: {overall_final:.2f} / {mean_IoU_final:.2f}')
    
    overall_topk= cum_I_topk * 100.0 / cum_U_topk
    mean_IoU_topk = torch.mean(torch.tensor(m_IoU_topk)) * 100.0

    f.write(f'\npure hybridgl (siglip2) top-k: {overall_topk:.2f} / {mean_IoU_topk:.2f}')  
    overall_final_topk = cum_I_final_topk * 100.0 / cum_U_final_topk
    mean_IoU_final_topk = torch.mean(torch.tensor(m_IoU_final_topk)) * 100.0

    f.write(f'\nhybridgl w/ spatial guidance (siglip2) to-k: {overall_final_topk:.2f} / {mean_IoU_final_topk:.2f}')

    div_score = torch.mean(torch.tensor(div_score))
    div_score_final = torch.mean(torch.tensor(div_score_final))

    f.write(f'\ndiversity scores: {div_score:.2f} / w/ sg {div_score_final:.2f}')

    f.close()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    with torch.no_grad():
        main(args, Height, Width)

