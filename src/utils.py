import os
BASE_DIR = os.environ.get("BASE_DIR", "/mnt/image-net-full/namin/ris")

import torch
import torch.nn as nn
import sys
import argparse
from typing import Optional, Tuple, List
from torch.nn.modules.module import Module
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.overrides import (has_torch_function, handle_torch_function, has_torch_function_variadic)
from torch.nn.functional import softmax, dropout

sys.path.append(os.path.join(BASE_DIR, 'third_party/FreeSOLO'))

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from freesolo import add_solo_config


# === Constants ===

MAIN_POS = ['NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM']

COLORS = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'orange', 'purple', 'brown', 'gray',
          'grey', 'teal', 'beige', 'maroon', 'lavender', 'turquoise', 'cyan', 'magenta', 'ivory', 'navy', 'gold',  
          'silver', 'bronze', 'dark', 'blurry', 'light']

SHAPES = [
  "triangle", "square", "pentagon", "hexagon", "heptagon", "octagon", "nonagon", "decagon",
  "dodecagon", "circle", "ellipse", "rectangle", "parallelogram", "rhombus", "trapezoid",
  "kite", "crescent", "sector", "annulus", "heart", "tetrahedron", "cube", "octahedron",
  "dodecahedron", "icosahedron", "sphere", "hemisphere", "cylinder", "cone", "pyramid",
  "prism", "torus", "ellipsoid", "frustum", "spiral", "helix", "lune", "lemniscate",
  "astroid", "cardioid", "hyperbola", "parabola"
]

RELATION_WORDS={"left", "west",\
                "right", "east",\
                "above", "north", "top", "back", "behind",\
                "below", "south", "under", "front",\
                "bigger", "larger", \
                "closer","smaller", "tinier", "further",\
                "inside", "within", "contained",\
                "who","what","which",\
                "middle"}

NULL_KEYWORDS = {"part", "image", "side", "picture", "half", "region", "section", "photo"}
LEFT_KEYWORDS = {"left", "west"}
RIGHT_KEYWORDS = {"right", "east"}
UP_KEYWORDS = {"above", "north", "top", "back", "behind"}
DOWN_KEYWORDS = {"below", "south", "under", "front"}
BIG_KEYWORDS = {"bigger", "larger", "closer"} 
SMALL_KEYWORDS = {"smaller", "tinier", "further", "smallest"}
WITHIN_KEYWORDS = {"inside", "within", "contained", "between"}

BASE_DIR = os.environ.get("BASE_DIR", "/mnt/image-net-full/namin/ris")

positional_prepositions = [
    "in", "on", "at", "inside", "outside", "above", "below", "under", "over",
    "between", "among", "beside", "behind", "next to", "near", "adjacent to",
    "in front of", "towards", "into", "onto", "out of", "away from", "of", "with", "right", "left"
]

# === Functions ===

def DFS_right(node, variables: list):
    if node.is_leaf() or variables[1] != '':
        return
    if node.label == 'NN':
        variables[1] = str(node.children[0])
        return
    for i in range(len(node.children)):
        idx = len(node.children) - i - 1
        DFS_right(node.children[idx], variables)

def DFS_left(node, variables: list):
    if node.is_leaf() or variables[1] != '':
        return
    if node.label == 'NP':
        variables[0] += 1
    now_find_np = variables[0]
    for child in node.children:
        DFS_left(child, variables)
    if node.label == 'NP' and variables[0] == now_find_np:
        # find rightmost NN
        DFS_right(node, variables)

def extract_noun_phrase_original(text, nlp, need_index=False):
    doc = nlp(text)
    variables = [0, '']
    DFS_left(doc.sentences[0].constituency, variables)
    agent = variables[1]
        
    if agent == '':
        for i in range(len(doc.sentences[0].words)):
            idx = len(doc.sentences[0].words) - i - 1
            if doc.sentences[0].words[idx].pos == 'NOUN' and doc.sentences[0].words[idx].text not in RELATION_WORDS and doc.sentences[0].words[idx].text not in ['half']:
                agent = doc.sentences[0].words[idx].text
                break
    if agent == '':
        agent = '[UNK]'
    agent_id = -1
    agents = []
    bef_noun_flag = True
    bef_rel_flag = True
    word_lst = []
    for word in doc.sentences[0].words:
        word_lst.append(word.text)
        if agent == word.text: 
            agent_id = word.id
            agents.append(word.text)
            bef_noun_flag = False
        else:
            if word.pos == 'NOUN' and bef_noun_flag:
                agents.append(word.text)
            if (word.text in COLORS or word.pos == 'NUM' or word.text in SHAPES) and word.text not in RELATION_WORDS: 
                agents.append(word.text)
                
    agent = ' '.join(word_lst[:agent_id])
      
    if need_index:
        return agents, agent, agent_id, len(doc.sentences[0].words)
    else:
        return agent

def mask_to_bbox_mask(binary_mask: torch.Tensor):
    """
    Convert a binary mask to its bounding-box mask.
    """
    binary_mask = binary_mask.bool()
    # Handle both (H, W) and (1, H, W) cases
    if binary_mask.dim() == 3:
        coords = torch.nonzero(binary_mask[0, :, :], as_tuple=False)
    else:
        coords = torch.nonzero(binary_mask, as_tuple=False)

    if coords.numel() == 0:
        return torch.zeros_like(binary_mask)

    y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
    x_min, x_max = coords[:, 1].min(), coords[:, 1].max()

    bbox_mask = torch.zeros_like(binary_mask)
    if binary_mask.dim() == 3:
        bbox_mask[:, y_min : y_max + 1, x_min : x_max + 1] = 1
    else:
        bbox_mask[y_min : y_max + 1, x_min : x_max + 1] = 1

    return bbox_mask


def extract_noun_phrase(text, nlp, need_index=False):
    doc = nlp(text)

    chunks = {}
    chunks_index = {}
    for chunk in doc.noun_chunks:
        for i in range(chunk.start, chunk.end):
            chunks[i] = chunk
            chunks_index[i] = (chunk.start, chunk.end)

    head = None
    for token in doc:
        if token.head.i == token.i:
            head = token.head
            break

    if head is None or head.i not in chunks:
        if head is not None:
            children = list(head.children)
            if children and children[0].i in chunks:
                head = children[0]
            else:
                if need_index:
                    return text, [], text
                else:
                    return text
        else:
            if need_index:
                return text, [], text
            else:
                return text

    head_noun = head.text
    head_index = chunks_index[head.i]
    head_index = [i for i in range(head_index[0], head_index[1])]

    sentence_index = [i for i in range(len(doc))]
    not_phrase_index = []
    for i in sentence_index:
        not_phrase_index.append(i) if i not in head_index else None

    head = chunks[head.i]
    if need_index:
        return head.text, not_phrase_index, head_noun
    else:
        return head.text


def extract_nouns(text, nlp, need_index=False):
    doc = nlp(text)
    noun_phrases = []
    nouns = []
    nouns_index = []
    head_noun = extract_noun_phrase(text, nlp)
    
    for chunk in doc.noun_chunks:
        if chunk.text == head_noun or chunk.root.text in RELATION_WORDS:
            continue
        noun_phrases.append(chunk.text)
        nouns_index.append((chunk.start, chunk.end))
        nouns.append(chunk.root.text)
    
    if need_index:
        return noun_phrases, nouns_index, nouns
    else:
        return noun_phrases, nouns


def extract_dir_phrase_desktop(text, nlp, need_index=False):
    dirflag = "none"
    diridx = 999
    deep2head = 999

    # text = text.lower()
    doc = nlp(text)

    for sentence in doc.sentences:
        for token in sentence.words:
            token_text = token.text
            # Stanza head index is 1-based; 0 means root (no head)
            head_i = token.head - 1  # convert to 0-based; -1 if root

            if token_text == "left" and head_i < deep2head:
                dirflag = "left"
                diridx = token.id - 1  # convert to 0-based
                deep2head = head_i
            elif token_text == "right" and head_i < deep2head:
                dirflag = "right"
                diridx = token.id - 1
                deep2head = head_i
            elif token_text in {"middle", "between"} and head_i < deep2head:
                dirflag = "middle"
                diridx = token.id - 1
                deep2head = head_i
            elif token_text in {"up", "top", "above"} and head_i < deep2head:
                dirflag = "up"
                diridx = token.id - 1
                deep2head = head_i
            elif token_text in {"down", "under", "bottom", "low"} and head_i < deep2head:
                dirflag = "down"
                diridx = token.id - 1
                deep2head = head_i

    if need_index:
        return dirflag, diridx
    else:
        return dirflag

def extract_dir_phrase(text, nlp, need_index=False):
    dirflag = "none"
    diridx = 999
    deep2head = 999
    doc = nlp(text)
    for token in doc:
        if token.text == "left" and token.head.i < deep2head:
            dirflag = "left"
            diridx = token.i
            deep2head = token.head.i
        elif token.text == "right" and token.head.i < deep2head:
            dirflag = "right"
            diridx = token.i
            deep2head = token.head.i
        elif token.text in {"middle", "between"} and token.head.i < deep2head:
            dirflag = "middle"
            diridx = token.i
            deep2head = token.head.i
        elif token.text in {"up", "top", "above"} and token.head.i < deep2head:
            dirflag = "up"
            diridx = token.i
            deep2head = token.head.i
        elif token.text in {"down", "under", "bottom", "low"} and token.head.i < deep2head:
            dirflag = "down"
            diridx = token.i
            deep2head = token.head.i

    if need_index:
        return dirflag, diridx
    else:
        return dirflag

def gen_dir_mask(dirflag, height, width, device=None):
    if dirflag == "left":
        a = torch.linspace(1, 0, width)
        pmask = a.expand(height, width)
    elif dirflag == "right":
        b = torch.linspace(0, 1, width)
        pmask = b.expand(height, width)
    elif dirflag == "middle":
        b1 = torch.linspace(0, 1, width // 2)
        b2 = torch.linspace(1, 0, width - width // 2)
        b = torch.cat([b1, b2])
        pmask = b.expand(height, width)
    else:
        pmask = torch.ones(height, width)
    
    if device:
        return pmask.to(device)
    else:
        return pmask

def extract_rela_word(text, nlp):
    noun_phrases, nouns = extract_nouns(text, nlp)
    if (set(nouns) & NULL_KEYWORDS):
        relaflag = "none"
    else:
        relaflag = "none"
        deep2head = 999
        doc = nlp(text)
        
        for token in doc:
            if token.text in LEFT_KEYWORDS and token.head.i < deep2head:
                relaflag = "left"
                deep2head = token.head.i
            elif token.text in RIGHT_KEYWORDS and token.head.i < deep2head:
                relaflag = "right"
                deep2head = token.head.i
            elif token.text in UP_KEYWORDS and token.head.i < deep2head:
                relaflag = "up"
                deep2head = token.head.i
            elif token.text in DOWN_KEYWORDS and token.head.i < deep2head:
                relaflag = "down"
                deep2head = token.head.i
            elif token.text in BIG_KEYWORDS and token.head.i < deep2head:
                relaflag = "big"
                deep2head = token.head.i
            elif token.text in SMALL_KEYWORDS and token.head.i < deep2head:
                relaflag = "small"
                deep2head = token.head.i
            elif token.text in WITHIN_KEYWORDS and token.head.i < deep2head:
                relaflag = "within"
                deep2head = token.head.i
    
    return relaflag

def relation_boxes(boxi, boxj, scorei, scorej, relaword):
    scoreout = 0

    if relaword == "none":
        scoreout = scorei
    elif relaword == "left":
        scoreout = scorei * scorej * ((boxi[0] + boxi[2] / 2) < (boxj[0] + boxj[2] / 2))
    elif relaword == "right":
        scoreout = scorei * scorej * ((boxi[0] + boxi[2] / 2) > (boxj[0] + boxj[2] / 2))
    elif relaword == "up":        
        scoreout = scorei * scorej * ((boxi[1] + boxi[3] / 2) < (boxj[1] + boxj[3] / 2))
    elif relaword == "down":        
        scoreout = scorei * scorej * ((boxi[1] + boxi[3] / 2) > (boxj[1] + boxj[3] / 2))
    elif relaword == "big":        
        scoreout = scorei * scorej * ((boxi[2] * boxi[3]) > (boxj[2] * boxj[3]))
    elif relaword == "small":        
        scoreout = scorei * scorej * ((boxi[2] * boxi[3]) < (boxj[2] * boxj[3]))
    elif relaword == "within":        
        x1 = max(boxi[0], boxj[0])
        x2 = max(x1, min(boxi[0] + boxi[2], boxj[0] + boxj[2]))
        y1 = max(boxi[1], boxj[1])
        y2 = max(y1, min(boxi[1] + boxi[3], boxj[1] + boxj[3]))
        scoreout = scorei * scorej * (x2 - x1) * (y2 - y1) / (boxi[2] * boxi[3])
    else :        
        scoreout = scorei

    return scoreout

def mask_iou_diversity(masks, div_score=[]):
    K = masks.shape[0]
    if K < 2:
        div_score.append(0.0)
        return div_score

    masks = masks.bool()
    iou_sum = 0.0
    count = 0

    for i in range(K):
        for j in range(i + 1, K):
            I = torch.logical_and(masks[i], masks[j]).sum()
            U = torch.logical_or(masks[i], masks[j]).sum()
            iou = 0.0 if U == 0 else I.float() / U.float()
            iou_sum += iou
            count += 1

    mean_iou = iou_sum / count
    div_score.append(1.0 - mean_iou)
    return div_score

def Compute_IoU(pred, target, cum_I, cum_U, mean_IoU=[], top_k=False, merge=False):
    if target.dtype != torch.bool:
        target = target.type(torch.bool).squeeze(0)
    if pred.dtype != torch.bool:
        pred = pred.type(torch.bool).squeeze(0)

    if top_k:
        I = torch.sum(torch.logical_and(pred, target), dim=(1, 2))
        U = torch.sum(torch.logical_or(pred, target), dim=(1, 2))
        return I, U
    
    if merge:
        pred = torch.sum(pred, dim=0).bool()

    I = torch.sum(torch.logical_and(pred, target))
    U = torch.sum(torch.logical_or(pred, target))
    
    if U == 0:
        iou = 0.0
    else:
        iou = I.float() / U.float()
    
    mean_IoU.append(iou.item())
    cum_I += I.item()
    cum_U += U.item()
    return mean_IoU, cum_I, cum_U

def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="configs/freesolo/freesolo_30k.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--resume", action="store_true", help="Whether to attempt to resume")
    parser.add_argument("--eval-only", action="store_false", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine")

    parser.add_argument("--delta", type=float, default=0.5, help="initial threshold")
    parser.add_argument("--alpha", type=float, default=0.5, help="proportion of spatial coherence")
    parser.add_argument("--layer", type=int, default=10, help="exit layer")
    parser.add_argument("--top_k", type=int, default=3, help="number of top clusters")
    parser.add_argument("--ten_percent", action="store_true", help="use first 10 percent of data")

    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    parser.add_argument('--clip_model', default='ViT-B/16')
    parser.add_argument('--visual_proj_path', default='./pretrain/')
    parser.add_argument('--dataset', default='refcocog')
    parser.add_argument('--split', default='val')
    parser.add_argument('--fusion_mode', default='G2L')
    parser.add_argument('--splitBy', default='umd')
    parser.add_argument('--unseen', action='store_true')
    parser.add_argument('--seen', action='store_true')
    parser.add_argument('--img_size', default=480, type=int)
    parser.add_argument('--refer_data_root', default=os.path.join(BASE_DIR, 'assets/ref_data/'))
    parser.add_argument('--show_results', action='store_true')

    return parser

def setup(args):
    cfg = get_cfg()
    add_solo_config(cfg)
    
    config_file = args.config_file
    if not os.path.exists(config_file):
        # Try finding it in third_party/FreeSOLO
        alt_config_file = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'FreeSOLO', config_file)
        if os.path.exists(alt_config_file):
            config_file = alt_config_file
        else:
            print(f"Warning: Config file {config_file} not found directly or in third_party/FreeSOLO")
            
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

# === MHA and Helpers ===

class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MHA, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.empty(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.empty(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None):
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    
    tgt_len, bsz, embed_dim = query.size()
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    q = q * scaling
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    if attn_mask is not None:
        attn_output_weights += attn_mask

    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

def _in_projection_packed(q, k, v, w, b=None):
    E = q.size(-1)
    if k is v:
        if q is k:
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is not None:
                b_q, b_kv = b.split([E, E * 2])
            else:
                b_q = b_kv = None
            return linear(q, w_q, b_q), *linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is not None:
            b_q, b_k, b_v = b.chunk(3)
        else:
            b_q = b_k = b_v = None
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(linear, (input, weight, bias), input, weight, bias=bias)
    return torch._C._nn.linear(input, weight, bias)
