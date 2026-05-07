import os
BASE_DIR = os.environ.get("BASE_DIR", "/mnt/image-net-full/namin/ris")

import warnings
warnings.filterwarnings("ignore")

import argparse
import clip
import torch
import os
import pandas as pd
from collections import Counter, defaultdict

Height, Width = 224, 224

import spacy
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from clip.simple_tokenizer import SimpleTokenizer
import tqdm
import cv2
from collections import defaultdict
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import gem

from dataset.dataset_refer_bert import ReferDataset
from model.backbone import clip_backbone, CLIPViTFM_Desktop
from src.templates import ext_dic 
from src.utils import default_argument_parser, setup, Compute_IoU, extract_rela_word, relation_boxes, extract_noun_phrase, extract_nouns, gen_dir_mask, extract_dir_phrase


def get_tensor_coordinates(index, size=7):
    """Convert a 1D index to 2D coordinates in a size x size tensor."""
    row_col = divmod(index, size)
    return row_col

def is_consecutive(coord1, coord2):
    """Check if two coordinates are consecutive in a row, column, or diagonal."""
    return abs(coord1[0] - coord2[0]) <= 1 and abs(coord1[1] - coord2[1]) <= 1

def custom_clustering(index_pairs, scores, threshold=0.95, size=14):
    """Perform custom clustering based on the given criteria."""
    clusters = {}
    cluster_id = 1
    prev_indices = []
    
    for (idx1, idx2), score_values in zip(index_pairs, scores):
        if score_values[1] <= threshold:
            break  
        
        if idx1 not in clusters:
            assigned_cluster = None
            for prev_idx in prev_indices:
                if is_consecutive(get_tensor_coordinates(prev_idx, size), get_tensor_coordinates(idx1, size)):
                    assigned_cluster = clusters[prev_idx]
                    break
            if assigned_cluster is not None:
                clusters[idx1] = assigned_cluster  # Assign cluster of the first consecutive previous index
            else:
                clusters[idx1] = cluster_id  # Assign new cluster
                cluster_id += 1
                
        prev_indices.append(idx1)
    
    # Post-process to merge isolated clusters if needed
    cluster_counts = Counter(clusters.values())
    unique_clusters = {cluster for cluster, count in cluster_counts.items() if count == 1}
    
    if len(unique_clusters) > 0:
        result = [key for key, value in clusters.items() if value in unique_clusters]
        for idx in result:
            idx_coord = get_tensor_coordinates(idx, size)
            for other_idx, other_cluster in clusters.items():
                if idx != other_idx and is_consecutive(idx_coord, get_tensor_coordinates(other_idx, size)):
                    clusters[idx] = clusters[other_idx]
                    break
    return clusters


def masks_to_boxes(masks):
    """
    Convert masks to bounding boxes. Assumes masks are binary (0 or 1).
    Args:
        masks (Tensor): [n, h, w] binary mask
    Returns:
        boxes (Tensor): [n, 4] in (x_min, y_min, x_max, y_max) format
    """
    n, h, w = masks.shape
    boxes = torch.zeros((n, 4), dtype=torch.float32)

    for i in range(n):
        mask = masks[i]
        y, x = torch.where(mask != 0)

        if len(x) == 0 or len(y) == 0:
            continue

        x_min = x.min().item()
        x_max = x.max().item()
        y_min = y.min().item()
        y_max = y.max().item()

        boxes[i] = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

    return boxes
    
                    
def main(args, Height, Width):
    assert args.eval_only, 'Only eval_only available!'
    cfg = setup(args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device available:', device)

    preprocess = gem.get_gem_img_transform()
    dataset = ReferDataset(args,
                           image_transforms=None,
                           target_transforms=None,
                           preprocessor = preprocess,
                           split=args.split)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
        
    from detectron2.config import get_cfg
    from detectron2.data.detection_utils import read_image
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.utils.logger import setup_logger
    import sys
    sys.path.append(os.path.join(BASE_DIR, 'third_party/IteRPrimE'))
    from mask2former import add_maskformer2_config
    sys.path.append(os.path.join(BASE_DIR, 'third_party/IteRPrimE/demo'))
    from predictor import VisualizationDemo

    def setup_cfg(args):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        return cfg

    class get_parser:
        config_file = os.path.join(BASE_DIR, "third_party/IteRPrimE/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
        opts = ['MODEL.WEIGHTS', os.path.join(BASE_DIR, "checkpoints/model_final_f07440.pkl")]
    
    new_args = get_parser()
    cfg = setup_cfg(new_args)
    demo = VisualizationDemo(cfg)

    mode = 'ViT' 
    assert (mode == 'Res') or (mode == 'ViT'), 'Specify mode(Res or ViT)'

    Model = clip_backbone(model_name='RN50').to(device) if mode == 'Res' else CLIPViTFM_Desktop(model_name='ViT-B/16').to(device)
    Model = Model.float()

    # import stanza
    # stanza.download('en')
    # nlp = stanza.Pipeline('en')
    nlp = spacy.load('en_core_web_lg')

    cum_I_ref, cum_U_ref =0, 0
    m_IoU_ref = []

    cum_I, cum_U =0, 0
    m_IoU = []

    real_cum_I, real_cum_U =0, 0
    real_m_IoU = []

    ref_real_cum_I, ref_real_cum_U =0, 0
    ref_real_m_IoU = []

    max_idx = 4
    v = 0.85 if args.dataset == 'refcocog' else 0.95
    r = 0.5

    tbar = tqdm.tqdm(data_loader)

    created_mask_num_lst = []
    our_mask_num_lst = []

    acc_lst = []
    gt_max_idx_lst = []

    threshold_lst = []

    softmax0 = torch.nn.Softmax(0)

    dim = 14 #14 # 7

    layer = 8 #8 # args.layer
    delta = 0.3 #args.delta #0.3 #args.delta
    alpha = 0.7 #args.alpha #0.7 #args.alpha
    top_k = args.top_k

    gem_model = gem.create_gem_model(
        model_name='ViT-B/16', pretrained='openai', device=device
    )
    
    save_dic = defaultdict(list)
    if args.ten_percent:
        data_num = int(len(data_loader)*0.1)
        
    for i, data in enumerate(tbar):
        if args.ten_percent and i >= data_num:
            break
  
        image, target, sentence_raw = data
        target = target.to(device)

        file_name = image['file_name'][0]
        assert os.path.exists(file_name)
        r_img = read_image(file_name, format="BGR") 
        _, _, pred_masks = demo.run_on_image(r_img)
        pred_boxes = masks_to_boxes(pred_masks)

        if len(pred_masks) == 0:
            print('No pred masks')
            continue

        original_imgs = torch.stack([T.Resize((height, width))(img.to(pred_masks.device)) for img, height, width in
                                     zip(image['image'], image['height'], image['width'])], dim=0) 
        resized_imgs = torch.stack([T.Resize((Height, Width))(img.to(pred_masks.device)) for img in image['image']], dim=0) 

        global_imgs = []
        local_imgs = []
        cropped_imgs = []

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(pred_masks.device)

        for pred_box, pred_mask in zip(pred_boxes.__iter__(), pred_masks):
            pred_mask, pred_box = pred_mask.type(torch.uint8), pred_box.type(torch.int)
            
            masked_image = original_imgs * pred_mask[None, None, ...] + (1 - pred_mask[None, None, ...]) * pixel_mean
    
            x1, y1, x2, y2 = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
            if x1 == x2:
                x2 = masked_image.shape[-1]
            if y1 == y2:
                y2 = masked_image.shape[-2]
            masked_image = TF.resized_crop(masked_image.squeeze(0), y1, x1, (y2 - y1), (x2 - x1), (Height, Width))
            cropped_imgs.append(masked_image)

        cropped_imgs = torch.stack(cropped_imgs, dim=0)

        with torch.no_grad():
            mask_features = Model.feature_map_masking(resized_imgs, pred_masks) if mode == 'Res' else Model(resized_imgs.to(device), pred_masks.to(device), masking_type='token_masking', masking_block=9)
    
            crop_features = Model.get_gloval_vector(cropped_imgs) if mode == 'Res' else Model(cropped_imgs.to(device), pred_masks=None, masking_type='crop')
    
            orig_visual_feature = v * mask_features + (1 - v) * crop_features
    
            visual_features = Model.get_gloval_vector(resized_imgs) if mode == 'Res' else Model(resized_imgs.to(device), pred_masks=None, masking_type='ours_local', masking_block=layer)

        for sentence in sentence_raw:
            temp_m_IoU = []
            
            sentence = sentence[0].lower()
            save_dic['sentence'].append(sentence)
            doc = nlp(sentence)
            sentence_for_spacy = []

            # stanza version
            # for i, token in enumerate(doc.sentences[0].words):
            #     if token.text == ' ':
            #         continue
            #     sentence_for_spacy.append(token.text)

            for i, token in enumerate(doc):
                if token.text == ' ':
                    continue
                sentence_for_spacy.append(token.text)

            orig_sentence_for_spacy = ' '.join(sentence_for_spacy)
            dirflag = extract_dir_phrase(orig_sentence_for_spacy, nlp, False)
            sentence_token = clip.tokenize(orig_sentence_for_spacy).to(device)
            
            noun_phrase, not_phrase_index, head_noun = extract_noun_phrase(orig_sentence_for_spacy, nlp, need_index=True)
            save_dic['positive_noun_phrase'].append(noun_phrase)
        
            noun_phrase_token = clip.tokenize(noun_phrase).to(device)
            noun_phrase_features = Model.get_text_feature(noun_phrase_token) if mode == 'Res' else Model.model.encode_text(noun_phrase_token)
            
            sentence_features = Model.get_text_feature(sentence_token) if mode == 'Res' else Model.model.encode_text(sentence_token)
            save_dic['img_dir'].append(image['file_name'])

            text_ensemble = r * sentence_features + (1-r) * noun_phrase_features

            orig_similarity = Model.calculate_similarity_score(orig_visual_feature, text_ensemble) if mode == 'Res' else Model.calculate_score(orig_visual_feature, text_ensemble)
            orig_max_index = torch.argmax(orig_similarity)
            orig_result_seg = pred_masks[orig_max_index]
            orig_IoU, _, _ = Compute_IoU(orig_result_seg, target, 0, 0, [])
            save_dic['orig_sim'].append(orig_similarity[orig_max_index].detach().cpu().numpy())
            sorted_indices = torch.argsort(orig_similarity.squeeze(), descending=True) 
            save_dic['sorted_indices'].append(sorted_indices.detach().cpu().numpy())

            other_noun_phrases, nouns = extract_nouns(orig_sentence_for_spacy, nlp)
            save_dic['other_noun_phrases'].append(other_noun_phrases)
            other_noun_features = torch.zeros(1, 512).to(device)
            cnt_other_nouns = 0
            for other_noun in other_noun_phrases:
                noun_token = clip.tokenize('a photo of '+other_noun).to(device)
                other_noun_features += Model.model.encode_text(noun_token)
                cnt_other_nouns += 1
            if cnt_other_nouns != 0:
                other_noun_features = other_noun_features / cnt_other_nouns
                
            orig_similarity_Neg = Model.calculate_score(orig_visual_feature, other_noun_features)
            
            orig_similarity = orig_similarity.squeeze()
            if len(orig_similarity.shape) < 1:
                orig_similarity = orig_similarity.unsqueeze(0)

            orig_similarity_Neg = orig_similarity_Neg.squeeze()
            if len(orig_similarity_Neg.shape) < 1:
                orig_similarity_Neg = orig_similarity_Neg.unsqueeze(0)

            similarity = Model.calculate_similarity_score(visual_features, text_ensemble) if mode == 'Res' else Model.calculate_score(visual_features, text_ensemble) 
            similarity = similarity.squeeze(0)

            chosen_sim = similarity[:, 0].reshape(dim, dim)
    
            width, height = pred_masks[0].shape

            # Flatten the tensor and get sorted indices in descending order
            sorted_indices = torch.argsort(chosen_sim.flatten(), descending=True)
            
            # Convert flat indices to 2D indices
            rows, cols = torch.div(sorted_indices, chosen_sim.shape[0], rounding_mode='floor'), sorted_indices % chosen_sim.shape[0]
            indices_2d = torch.stack((rows, cols), dim=1) 

            test_index_pairs = []
            test_scores = []
            for idx in range(indices_2d.shape[0]-1):
                i, j, i_next, j_next = indices_2d[idx][0].item(), indices_2d[idx][1].item(), indices_2d[idx+1][0].item(), indices_2d[idx+1][1].item()
                test_index_pairs.append((dim*i+j, dim*(i_next)+(j_next)))
                
                sim_ij = F.cosine_similarity(visual_features[0][dim*i+j], visual_features[0][dim*(i_next)+(j_next)], dim=0)
                text_sim_ij = chosen_sim[i,j] / 100
                text_sim_ij_next = chosen_sim[i_next,j_next] / 100
                # print(7*i+j, 7*(i_next)+(j_next), sim_ij.item(), text_sim_ij.item(), text_sim_ij_next.item())
                test_scores.append((sim_ij, text_sim_ij, text_sim_ij_next))

            # Find cluster dict until it becomes more than 1
            cluster_dict = {}
            threshold = delta
            while len(cluster_dict) <= 0:
                threshold -= 0.05
                cluster_dict = custom_clustering(test_index_pairs, test_scores, threshold, dim)
            threshold_lst.append(threshold)

            # print('Number of cluster found:', len(cluster_dict), '(Threshold: ', threshold, ')')
            save_dic['num_clusters'].append(len(cluster_dict))
            assert len(cluster_dict) > 0
            # Initialize a 7x7 tensor (matrix)
            tensor_shape = (dim, dim)
            tensor = torch.zeros(tensor_shape, dtype=int)
            
            # Fill the tensor with the cluster number at the corresponding indices
            for idx, cluster in cluster_dict.items():
                row, col = divmod(idx, tensor_shape[1])
                tensor[row, col] = cluster
            
            # Create binary masks for each cluster
            for i,cluster in enumerate(set(cluster_dict.values())):
                mask = (tensor == cluster).int().to(chosen_sim.device)
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),size=(width,height), mode='bilinear', align_corners=True)[0]
                if i == 0:
                    stacked_mask = mask
                else:
                    stacked_mask = torch.cat((stacked_mask, mask), 0)
            stacked_mask = stacked_mask.to(pred_masks.device)[:top_k]

            best_indices = [orig_max_index.item()]
            best_IoUs = [torch.tensor(1.)] 
            for j in range(stacked_mask.shape[0]):
                temp_m_IoU2 = []
                crit_scores = []
                for i in range(pred_masks.shape[0]):
                    cur_IoU, _, _ = Compute_IoU(stacked_mask[j], pred_masks[i], 0, 0, [])
                    temp_m_IoU2.append(cur_IoU)
                    den = pred_masks[i].sum()
                    num = (pred_masks[i]*stacked_mask[j]).sum()
                    crit_scores.append(1 + num/den + stacked_mask[j].sum()/stacked_mask.sum())
                    
                max_idx_per_mask = temp_m_IoU2.index(max(temp_m_IoU2))
                best_indices.append(max_idx_per_mask) 
                best_IoUs.append(crit_scores[max_idx_per_mask])

            best_IoUs = [it.item() for it in best_IoUs]
            temp_real_idx = best_IoUs.index(max(best_IoUs))
            save_dic['best_indices'].append(torch.tensor(best_indices).detach().cpu().numpy())
            save_dic['best_overlapping_IoUs'].append(torch.tensor(best_IoUs).detach().cpu().numpy())

            partial_pred_masks = pred_masks[torch.tensor(best_indices)]
            partial_pred_boxes = pred_boxes[torch.tensor(best_indices)] 
            created_mask_num_lst.append(partial_pred_masks.shape[0])

            concat_partial_pred_masks = partial_pred_masks 
            our_mask_num_lst.append(partial_pred_masks.shape[0])

            real_temp_m_IoU = []
            real_save_cum_I = []
            real_save_cum_U = []
            for best_idx in range(concat_partial_pred_masks.shape[0]):
                temp_cum_I, temp_cum_U = 0, 0
                real_temp_m_IoU, temp_cum_I, temp_cum_U = Compute_IoU(concat_partial_pred_masks[best_idx], target, temp_cum_I, temp_cum_U, real_temp_m_IoU)
                real_save_cum_I.append(temp_cum_I)
                real_save_cum_U.append(temp_cum_U)

            temp_real_temp_m_IoU = [r for r in real_temp_m_IoU]
            sorted_indices = np.argsort(temp_real_temp_m_IoU)
            
            real_temp_m_IoU_s, gt_max_idx = max(real_temp_m_IoU), real_temp_m_IoU.index(max(real_temp_m_IoU))
         
            real_cum_I += real_save_cum_I[gt_max_idx]
            real_cum_U += real_save_cum_U[gt_max_idx]
        
            real_m_IoU.append(real_temp_m_IoU_s)
            save_dic['upper_IoU'].append(real_temp_m_IoU_s) #.detach().cpu().numpy())
            gt_max_idx_lst.append(gt_max_idx)
            save_dic['upper_IoU_idx'].append(gt_max_idx)
            save_dic['upper_IoU_real_idx'].append(best_indices[gt_max_idx])

            # Spatial Relationship Guidance 
            relaflag = extract_rela_word(orig_sentence_for_spacy, nlp)
            score_clip = softmax0(orig_similarity)
            score_clip_Neg = softmax0(orig_similarity_Neg)
            k1 = min(top_k, score_clip.shape[0]) 
            k2 = min(6, score_clip_Neg.shape[0])
            
            k1 = min(k1, concat_partial_pred_masks.shape[0])
            maxidxs = best_indices[:k1]
            
            save_dic['maxidxs'].append([ii for ii in maxidxs])
            _, maxNegidxs = torch.topk(score_clip_Neg.view(-1),k=k2) 
            save_dic['maxNegidxs'].append([ii.item() for ii in maxNegidxs])
            
            save_dic['num_pred_masks'].append(pred_masks.shape[0])
            
            topscores = np.zeros(k1) 
            boxes = pred_boxes #.tensor
            # reranking the score
            if len(nouns)==0: 
                for i,idx_i in enumerate(maxidxs): 
                    for idx_j in maxidxs:   
                        s = relation_boxes(boxes[idx_i], 
                                           boxes[idx_j], 
                                           score_clip[idx_i], 
                                           score_clip[idx_j],
                                           relaflag)
                        topscores[i]=topscores[i] + s
            else:
                for i,idx_i in enumerate(maxidxs): 
                    for idx_j in maxNegidxs:   
                        s = relation_boxes(boxes[idx_i], 
                                           boxes[idx_j],
                                           score_clip[idx_i],
                                           score_clip_Neg[idx_j],
                                           relaflag)
                        topscores[i]=topscores[i] + s
                        
            save_dic['orig_orig_topscores'].append([score_clip[ii].item() for ii in maxidxs])
            save_dic['orig_topscores'].append(topscores.tolist())
            topscores = torch.Tensor(topscores).to(device)
            topscores = softmax0(topscores)
            our_final_idx = maxidxs[torch.argmax(topscores)] 

            try:
                save_dic['orig_orig_IoU_idx'].append(maxidxs[0].detach().cpu().numpy())
            except:
                save_dic['orig_orig_IoU_idx'].append(maxidxs[0])
            try:
                save_dic['orig_IoU_idx'].append(our_final_idx.detach().cpu().numpy())
            except:
                save_dic['orig_IoU_idx'].append(our_final_idx)
                
            # ########  Spatial Coherence Guidance ########
            score_gem_list=[]

            imgattn = gem_model(image['tensor_img'].to(device), [noun_phrase])[0]
            imgattn = T.Resize((int(image['height']), int(image['width'])))(imgattn)[0]
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
                
            for pred_mask in pred_masks:
                pred_mask = pred_mask.type(torch.uint8)
                score_gemtmp = (imgattn * (2-black) * pred_mask/(pred_mask.sum())).sum() - (imgattn * black * (1 - pred_mask) / ((1 - pred_mask).sum())).sum()
                score_gem_list.append(torch.Tensor([score_gemtmp]))
            score_gem = torch.stack(score_gem_list,dim=0)
            score_gem=score_gem.to(device)

            for idx_i in range(k1):      
                topscores[idx_i]=topscores[idx_i] * (1 - alpha) + alpha * score_gem[maxidxs[idx_i]][0]
            save_dic['new_topscores'].append(topscores.tolist())
            our_final_idx = maxidxs[torch.argmax(topscores)]

            try:
                save_dic['our_IoU_idx'].append(our_final_idx.detach().cpu().numpy())
            except:
                save_dic['our_IoU_idx'].append(our_final_idx)
            
            comp_max_idx = our_final_idx 
            our_final_mask = pred_masks[comp_max_idx]  
            m_IoU, cum_I, cum_U = Compute_IoU(our_final_mask, target, cum_I, cum_U, m_IoU)

            our_IoU = m_IoU[-1]
            save_dic['our_IoU'].append(our_IoU)
            print(our_IoU, flush=True)
            

    df = pd.DataFrame(save_dic)
    df.to_csv(f"analysis_2_final_{delta}_{layer}_{alpha}_{args.dataset}_{args.split}_{args.splitBy}_2026.csv", index=False)

                    
    f = open('./result_log/results_newmask2former_vitb16.txt', 'a')
    f.write(f'\n\n CLIP Model (ViT-B/16): {mode} layer: {layer} delta: {delta} alpha: {alpha} topk: {top_k}'
            f'\nDataset: {args.dataset} / {args.split} / {args.splitBy}'
            f'\nOverall IoU / mean IoU')

    overall = cum_I * 100.0 / cum_U
    mean_IoU = torch.mean(torch.tensor(m_IoU)) * 100.0

    f.write(f'\n{overall:.2f} / {mean_IoU:.2f}')

    real_overall = real_cum_I * 100.0 / real_cum_U
    real_mean_IoU = torch.mean(torch.tensor(real_m_IoU)) * 100.0

    f.write(f'\n{real_overall:.2f} / {real_mean_IoU:.2f}')

    f.write(f'Threshold {np.mean(threshold_lst):.2f} / {np.std(threshold_lst):.2f}')

    f.write(f'Number of selected candidates: {np.mean(created_mask_num_lst):.2f} / {np.std(created_mask_num_lst):.2f}')
    f.write(f'Number of created clusters: {np.mean(our_mask_num_lst):.2f} / {np.std(our_mask_num_lst):.2f}')



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    opts = ['OUTPUT_DIR', 'training_dir/FreeSOLO_pl', 'MODEL.WEIGHTS', os.path.join(BASE_DIR, 'checkpoints/model_final_f07440.pkl')]
    args.opts = opts
    print(args.opts)
    main(args, Height, Width)
