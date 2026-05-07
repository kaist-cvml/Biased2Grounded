import os
BASE_DIR = os.environ.get("BASE_DIR", "/mnt/image-net-full/namin/ris")

from pydoc import text
from turtle import hideturtle, mainloop
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import spacy
import numpy as np
import tqdm
import cv2
import time
import pandas as pd
from transformers import AutoModel, AutoProcessor
from torch.cuda.amp import autocast
from PIL import Image
from transformers.image_utils import load_image
import os
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional import to_pil_image

from dataset.dataset_refer_bert import ReferDataset
from model.backbone import CLIPViTFM
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
    # dataset = MultilingualDataset(preprocessor = preprocess, dataset=args.dataset, split=args.split)
    dataset = ReferDataset(args,
                           image_transforms=None,
                           target_transforms=None,
                           preprocessor = preprocess,
                           split=args.split,
                           is_multilingual=True)
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
    total_inference_time = 0.0
    total_samples = 0
    tbar = tqdm.tqdm(data_loader)

    ########################## load data ##########################

    results_per_instance = []

    # Create directory for saving masks
    mask_save_dir = f'./result_log/masks_multilingual_{args.dataset}_{args.split}'
    os.makedirs(mask_save_dir, exist_ok=True)

    for i, data in enumerate(tbar):
        orig_i = i
        # if i == 10:
        #     break
        
        image, target, sentence_raw, lang_dic = data
        if lang_dic:
            image_path = data[0]['file_name'][0]
            
            # --- Visual Feature Extraction ---
            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            v_start_time = time.perf_counter()

            target = target.to(device)
            target_new = mask_to_bbox_mask(target)
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

            if device == 'cuda':
                torch.cuda.synchronize()
                v_peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            else:
                v_peak_mem = 0.0
            v_elapsed = time.perf_counter() - v_start_time

            languages = list(lang_dic.keys())
            num_sentences = len(lang_dic[languages[0]])

            # --- Text Feature Extraction ---
            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            t_start_time = time.perf_counter()

            for iii in range(num_sentences):
                main_embs = []
                noun_embs = []
                other_embs = []

                for lang in languages:
                    sentence = lang_dic[lang][iii][0].lower()
                    # print(lang, sentence)
                    doc = nlp(sentence)

                    sentence_for_spacy = [
                        token.text for token in doc if token.text != ' '
                    ]
                    sentence_for_spacy = ' '.join(sentence_for_spacy)

                    dirflag = extract_dir_phrase(sentence_for_spacy, nlp, False)
                    # visual_feature = hybrid_features
                    
                    # Use SigLIP22 tokenizer instead of CLIP
                    sentence_token = Model.tokenizer(sentence_for_spacy, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
                    noun_phrase, not_phrase_index, head_noun = extract_noun_phrase(sentence_for_spacy, nlp, need_index=True)
                    
                    noun_phrase_token = Model.tokenizer(noun_phrase, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
                    
                    sent_emb = Model.model.text_model.embeddings(**sentence_token)          # (B, N, D)
                    noun_emb = Model.model.text_model.embeddings(**noun_phrase_token)      # (B, N, D)
                    
                    noun_phrases, nouns = extract_nouns(sentence_for_spacy, nlp)
                    # SigLIP22 uses 768-dimensional embeddings instead of 512
                    other_noun_emb = torch.zeros(1, Model.model.config.text_config.hidden_size).to(device)
                    cnt_other_nouns = 0
                    for other_noun in noun_phrases:
                        noun_token = Model.tokenizer('This is a photo of '+other_noun, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
                        other_noun_emb += Model.model.get_text_features(**noun_token)
                        cnt_other_nouns += 1
                    if cnt_other_nouns != 0:
                        other_noun_emb = other_noun_emb / cnt_other_nouns

                    main_embs.append(sent_emb)
                    noun_embs.append(noun_emb)
                    other_embs.append(other_noun_emb)

                scores = []

                layers = Model.model.text_model.encoder.layers
                num_layers = len(layers)
                start_idx = 0 #num_layers // 2  # last half

                for i, layer in enumerate(layers[start_idx:], start=start_idx):
                    layer_sent = []
                    layer_noun = []

                    for ll in range(len(main_embs)):
                        sent_l = layer(main_embs[ll], attention_mask=None)
                        noun_l = layer(noun_embs[ll], attention_mask=None)
                        layer_sent.append(sent_l[0].squeeze(0))  # (N, D)
                        layer_noun.append(noun_l[0].squeeze(0))
                    
                    sim_sent_total = 0
                    sim_noun_total = 0
                    count = 0

                    for a in range(len(layer_sent)):
                        for b in range(a+1, len(layer_sent)):
                            sim_sent_total += F.cosine_similarity(
                                layer_sent[a], layer_sent[b], dim=-1
                            ).mean()
                            
                            sim_noun_total += F.cosine_similarity(
                                layer_noun[a], layer_noun[b], dim=-1
                            ).mean()
                            
                            count += 1

                    score = (sim_sent_total + sim_noun_total) / count
                    scores.append(score.item())

                relative_idx = torch.tensor(scores).argmax().item()
                merge_layer = start_idx + relative_idx
                
                # initialize
                current_sent = main_embs
                current_noun = noun_embs

                for i, layer in enumerate(Model.model.text_model.encoder.layers):
                    if i == merge_layer:
                        break

                    for l in range(len(current_sent)):
                        current_sent[l] = layer(current_sent[l], attention_mask=None)[0]
                        current_noun[l] = layer(current_noun[l], attention_mask=None)[0]

                merged_sent = torch.stack(current_sent).mean(dim=0)
                merged_noun = torch.stack(current_noun).mean(dim=0)

                merged = (merged_sent + merged_noun) / 2

                for i in range(merge_layer, len(Model.model.text_model.encoder.layers)):
                    merged = Model.model.text_model.encoder.layers[i](merged, attention_mask=None)[0]

                merged = Model.model.text_model.final_layer_norm(merged)
                text_ensemble = Model.model.text_model.head(merged[:, -1, :]) # (1, 768)


                # score_clip = Model.calculate_score(hybrid_features, text_ensemble)
                

                other_noun_features = torch.stack(other_embs)          # [N, D]
                other_noun_features = other_noun_features.mean(dim=0)        # [1, D]                
                other_noun_features = other_noun_features / other_noun_features.norm(dim=-1, keepdim=True)
                    
                # score_clip_Neg = Model.calculate_score(hybrid_features, other_noun_features)

            if device == "cuda":
                torch.cuda.synchronize()
                t_peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            else:
                t_peak_mem = 0.0
            t_elapsed = time.perf_counter() - t_start_time

            total_inference_time += (v_elapsed + t_elapsed)
            total_samples += 1   

            results_per_instance.append({
                "index": orig_i,
                "image_path": image_path,
                "visual_elapsed": float(v_elapsed),
                "visual_peak_gpu_mem_MB": float(v_peak_mem),
                "text_elapsed": float(t_elapsed),
                "text_peak_gpu_mem_MB": float(t_peak_mem)
            })
              

    df = pd.DataFrame(results_per_instance)
    df.to_csv(f"./result_log/per_instance_multilingual_siglip2_{args.dataset}_{args.split}_{i}_centroid_textlayer_cost.csv", index=False)
    print("Saved per-instance CSV:", f"./result_log/per_instance_multilingual_siglip2_{args.dataset}_{args.split}_{i}_centroid_textlayer_cost.csv")



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    with torch.no_grad():
        main(args, Height, Width)

