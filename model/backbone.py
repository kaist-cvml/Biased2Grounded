import torch
import torch.nn as nn
import clip
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import copy
from transformers import AutoModel, AutoProcessor
import open_clip
from open_clip import create_model_from_pretrained, create_model_and_transforms, get_tokenizer

# === From backbone.py ===

class CLIPViTFM(nn.Module):  
    def __init__(self, model_name='ViT-B/16', size=224):
        super().__init__()
        
        self.use_siglip = 'siglip' in model_name.lower()
        self.size = size

        if self.use_siglip:
            # Load SigLIP2-base-patch16-384
            if 'siglip2' in model_name.lower():
                model_id = "google/siglip2-base-patch16-384"
            elif 'siglip' in model_name.lower():
                model_id = "google/siglip-base-patch16-384"
            self.model = AutoModel.from_pretrained(model_id)

            self.tokenizer = AutoProcessor.from_pretrained(model_id).tokenizer
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # SigLIP2-base-patch16-384 configuration
            self.last_layer = 10  # SigLIP base has 12 layers (0-11), use layer 11
            self.num_heads = 12   # SigLIP base has 12 attention heads
            self.embed_dim = 768  # SigLIP base embedding dimension
            
            # Access vision model for compatibility
            self.vision_model = self.model.vision_model

            self.pos_embed = self.vision_model.embeddings.position_embedding.weight
        else:
            # Original CLIP loading
            if model_name == 'ViT-B/32':
                self.last_layer = 10
                self.num_heads = 12
            elif model_name == 'ViT-B/16':
                self.last_layer = 10
                self.num_heads = 12

            self.model, _ = clip.load(model_name)


    @property
    def device(self):
        if self.use_siglip:
            return next(self.model.parameters()).device
        else:
            return self.model.visual.conv1.weight.device

    @property
    def dtype(self):
        if self.use_siglip:
            return next(self.model.parameters()).dtype
        else:
            return self.model.visual.conv1.weight.dtype


    def encode_text_attention(self, noun_phrase_token, sentence_token, r, fusion_layer=9, is_siglip=False):
        if not is_siglip:
            text_encoder = self.model.transformer
            
            # Step 1: Embed inputs
            np_embed = self.model.token_embedding(noun_phrase_token).type(self.dtype) + self.model.positional_embedding.type(self.dtype)
            sent_embed = self.model.token_embedding(sentence_token).type(self.dtype) + self.model.positional_embedding.type(self.dtype)
        
            # [1, 77, 512] → [77, 1, 512]
            np_embed = np_embed.permute(1, 0, 2)
            sent_embed = sent_embed.permute(1, 0, 2)
        
            # Step 2: Process through transformer layers
            for block_idx, resblock in enumerate(text_encoder.resblocks):
                if block_idx < fusion_layer:
                    np_embed = resblock(np_embed)
                    sent_embed = resblock(sent_embed)
        
                elif block_idx == fusion_layer:
                    # Step 3: Fuse the self-attention outputs
                    np_ln = resblock.ln_1(np_embed) # [77, 1, 512]
                    sent_ln = resblock.ln_1(sent_embed)
        
                    np_attn = resblock.attn(np_ln, np_ln, np_ln)[0] # [77, 1, 512]
                    sent_attn = resblock.attn(sent_ln, sent_ln, sent_ln)[0]

                    # np_attn = np_attn + np_embed
                    # sent_attn = sent_attn + sent_embed
        
                    # Fuse attention embeddings (e.g., average or concat + proj)
                    fused_attn = r * sent_attn + (1-r)  * np_attn
        
                    # Add residual
                    fused_embed = 0.5 * (np_embed + sent_embed) + fused_attn
        
                    # MLP part
                    fused_embed = fused_embed + resblock.mlp(resblock.ln_2(fused_embed)) # [77, 1, 512]
        
                else:
                    # Use fused representation for the remaining layers
                    fused_embed = resblock(fused_embed)
        
            # Step 4: Final projection
            fused_embed = fused_embed.permute(1, 0, 2)  # [1, seq_len, dim]
            fused_embed = self.model.ln_final(fused_embed).type(self.dtype)
        
            # Step 5: Extract CLS or EOS token
            final_embedding = fused_embed[torch.arange(fused_embed.shape[0]), noun_phrase_token.argmax(dim=-1)]
            return final_embedding @ self.model.text_projection
        else:
            text_encoder = self.model.text_model

            # Step 1: Embed inputs
            noun_phrase_token_input_ids, sentence_token_input_ids = noun_phrase_token['input_ids'], sentence_token['input_ids']
            # input_shape = noun_phrase_token_input_ids.size()
            # noun_phrase_token_input_ids = noun_phrase_token_input_ids.view(-1, input_shape[-1])
            # input_shape = sentence_token_input_ids.size()
            # sentence_token_input_ids = sentence_token_input_ids.view(-1, input_shape[-1])

            noun_phrase_token_attn_mask, sentence_token_attn_mask = None, None
            
            noun_position_ids = text_encoder.embeddings.position_ids[:, :noun_phrase_token_input_ids.shape[-1]]
            np_embed = text_encoder.embeddings.token_embedding(noun_phrase_token_input_ids).type(self.dtype) + text_encoder.embeddings.position_embedding.type(self.dtype)(noun_position_ids)
            sent_position_ids = text_encoder.embeddings.position_ids[:, :sentence_token_input_ids.shape[-1]]
            sent_embed = text_encoder.embeddings.token_embedding(sentence_token_input_ids).type(self.dtype) + text_encoder.embeddings.position_embedding.type(self.dtype)(sent_position_ids)
            # torch.Size([1, 64, 768])
        
            # Step 2: Process through transformer layers
            for block_idx, resblock in enumerate(text_encoder.encoder.layers):
                if block_idx < fusion_layer:
                    np_embed = resblock(np_embed, attention_mask=noun_phrase_token_attn_mask)
                    sent_embed = resblock(sent_embed, attention_mask=sentence_token_attn_mask)
        
                elif block_idx == fusion_layer:
                    # Step 3: Fuse the self-attention outputs
                    np_ln = resblock.layer_norm1(np_embed) # [77, 1, 512]
                    sent_ln = resblock.layer_norm1(sent_embed) # torch.Size([64, 1, 768])
        
                    np_attn = resblock.self_attn(np_ln, attention_mask=noun_phrase_token_attn_mask)[0] # [77, 1, 512]
                    sent_attn = resblock.self_attn(sent_ln, attention_mask=sentence_token_attn_mask)[0]
        
                    # Fuse attention embeddings (e.g., average or concat + proj)
                    fused_attn = r * sent_attn + (1-r) * np_attn
        
                    # Add residual
                    fused_embed = sent_embed + fused_attn
        
                    # MLP part
                    fused_embed = fused_embed + resblock.mlp(resblock.layer_norm2(fused_embed)) # [77, 1, 512]
        
                else:
                    # Use fused representation for the remaining layers
                    fused_embed = resblock(fused_embed, attention_mask=sentence_token_attn_mask)
        
            # Step 4: Final projection
            last_hidden_state = text_encoder.final_layer_norm(fused_embed).type(self.dtype)
            pooled_output = last_hidden_state[:, -1, :]
            pooled_output = text_encoder.head(pooled_output)

            # Step 5: Extract the pooler output
            return pooled_output


    def text_masking_feature(self, text, masking_index=[], masking_block=11):
        text_encoder = self.model.transformer
        masking_index = [i+1 for i in masking_index] # because start token
 
        x = self.model.token_embedding(text).type(self.dtype) # [1,77,512]
        x = x + self.model.positional_embedding.type(self.dtype) # [1,77,512]
        x = x.permute(1, 0, 2) # [77, 1, 512]

        for block_idx, resblock in enumerate(text_encoder.resblocks): # last block idx [11, 11, 23]
            if block_idx >= masking_block:
                if masking_index:
                    x[masking_index] = 0
                    x = resblock(x) 
                else:
                    x = self._call_transformer_block(resblock, x)
            else:
                x = resblock(x)
 
        x = x.permute(1, 0, 2) # [1, 77, 512]
        x = self.model.ln_final(x).type(self.dtype) # [1, 77, 512]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection # [1, 512]

        return x
    
    def text_feature(self, text):
        text_encoder = self.model.transformer 
        x = self.model.token_embedding(text).type(self.dtype) # [1,77,512]
        x = x + self.model.positional_embedding.type(self.dtype) # [1,77,512]
        x = x.permute(1, 0, 2) # [77, 1, 512]
 
        x = text_encoder.resblocks(x)

        x = x.permute(1, 0, 2) # [1, 77, 512]
        x = self.model.ln_final(x).type(self.dtype) # [1, 77, 512]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection # [1, 512]
        
        return x
    


    def calculate_score(self, image_features, text_features, visual_norm_dim=1):
        # image_features = [N,512] or [N,768] for SigLIP
        # text_feature = [1,512] or [1,768] for SigLIP

        image_features = image_features / image_features.norm(dim=visual_norm_dim, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
    
     
        return logits_per_image # [N, 1]

    def upsample_pos_emb(self, emb):
        # upsample the pretrained embedding for higher resolution
        # emb size NxD
        first = emb[:1, :]
        emb = emb[1:, :]
        N, D = emb.size(0), emb.size(1)
        n = int(np.sqrt(N))
        assert n * n == N

        emb = emb.permute(1, 0)
        emb = emb.view(1, D, n, n).contiguous()
        emb = F.upsample(emb, size=(self.size), mode='bilinear',
                         align_corners=None)
        emb = emb.view(D, -1).contiguous()
        emb = emb.permute(1, 0)
        emb = torch.cat([first, emb], 0)
        emb = nn.parameter.Parameter(emb.half())
        return emb

    def make_attn_mask(self, pred_masks, size=None, use_siglip=False):
        # pred_masks =  [46, H, W], Torch.bool
        if size is not None:
            pred_masks = TF.resize(pred_masks, size=(size, size))  # [46,7,7]

        if use_siglip:
            # pred_tokens = pred_masks.flatten(1) 
            # pred_tokens_bool = pred_tokens > 0.5         # bool [B, HW]

            # attn_mask = torch.where(
            #     pred_tokens_bool,
            #     torch.zeros_like(pred_tokens, dtype=self.dtype),                 # keep
            #     torch.full_like(pred_tokens, float("-inf"), dtype=self.dtype),   # block
            # )

            # attn_masks = attn_mask[:, None, None, :]       # [B, 1, 1, HW]
            return pred_masks*1.0

        else:
            N, H, W = pred_masks.size()
            attn_masks = torch.ones((N * self.num_heads, H*W+1, H*W+1), dtype=torch.bool).to(self.device)
            attn_masks[:, 0, 1:] = pred_masks[:,None,:,:].expand(-1,self.num_heads,-1,-1).reshape(N * self.num_heads, -1)
            return ~attn_masks
    
    def _call_transformer_block(self, block, x, attn_mask=None):
        """Helper function to call transformer blocks for both CLIP and SigLIP"""
        if self.use_siglip:
            # SigLIP encoder layers have a different interface
            # They expect input in [B, N, D] format, not [N, B, D]
            if x.dim() == 3 and x.shape[0] != x.shape[1]:  # [N, B, D] format
                x = x.permute(1, 0, 2)  # Convert to [B, N, D]
            # SigLIP doesn't support attn_mask in the same way, so we ignore it for now
            # This might need adjustment based on actual SigLIP implementation
            if self.use_siglip:
                output = block(x, attention_mask=attn_mask)
            else:
                output = block(x)
            if hasattr(output, 'last_hidden_state'):
                output = output.last_hidden_state
            # Convert back to [N, B, D] if needed
            if output.shape[1] == x.shape[0]:  # Check if we need to permute back
                output = output.permute(1, 0, 2)
            return output
        else:
            # CLIP resblock
            if attn_mask is not None:
                return block(x, attn_mask=attn_mask)
            else:
                return block(x)
    
    def forward(self, local_imgs, global_imgs, pred_masks, masking_block=None, fusion_mode='G2L'):
        if masking_block is None:
            masking_block = self.last_layer

        if self.use_siglip:
            vit = self.vision_model
        else:
            vit = self.model.visual

        x = local_imgs.type(self.dtype)
        

        if fusion_mode == 'crop': # [1, 512] or [1, 768]
            if self.use_siglip:
                # SigLIP forward
                outputs = vit(pixel_values=x)
                return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0, :]
            else:
                x = vit(x)
                return x[:, 0, :]

        if not self.use_siglip:
            x = vit.conv1(x)  # shape = [*, width, grid, grid]

            x = x.reshape(x.shape[0], x.shape[1], -1)

            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                         dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

            x = x + vit.positional_embedding.to(x.dtype)
            x = vit.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
        ####################### x2 ####################################
        if global_imgs is not None:
            if self.use_siglip:
                x2 = global_imgs.type(self.dtype)

                local_outputs = vit(pixel_values=x)

                attn_mask = self.make_attn_mask(pred_masks, size=384, use_siglip=True)
                # if attn_mask.shape[0] > 515:
                #     attn_mask = attn_mask[:515, :, :]
                # print(x2.shape, attn_mask.shape)
                global_outputs = vit(pixel_values=x2*attn_mask.unsqueeze(1)) 

                return 0.5*global_outputs.pooler_output + 0.5*local_outputs.pooler_output
                
            else:
                x2 = global_imgs.type(self.dtype)
                x2 = vit.conv1(x2)  # shape = [*, width, grid, grid]
                x2 = x2.reshape(x2.shape[0], x2.shape[1], -1)
                x2 = x2.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x2 = torch.cat([vit.class_embedding.to(x2.dtype) + torch.zeros(x2.shape[0], 1, x2.shape[-1],
                                                                            dtype=x2.dtype, device=x2.device), x2], dim=1)  # shape = [*, grid ** 2 + 1, width]

                x2 = x2 + vit.positional_embedding.to(x2.dtype)
                x2 = vit.ln_pre(x2)

                x2 = x2.permute(1, 0, 2)  # NLD -> LND

        if not self.use_siglip:
            L, N, D = x[1:, :, :].size(0), x.size(1), x.size(2)
            size = int(np.sqrt(L))
            assert size * size == L

        pred_masks = TF.resize(pred_masks.type(torch.float32), (size, size))
        if fusion_mode == 'token_masking':
            transformer_blocks = vit.encoder.layers if self.use_siglip else vit.transformer.resblocks
            for block_idx, resblock in enumerate(transformer_blocks):
                if block_idx >= masking_block:
                    cls = x[:1,:,:]
                    x = x[1:,:,:]
                    x = x.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]

                    x = x.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]

                    x = torch.mul(x, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                    N = x.size(0)
                    x = x.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]

                    x = x.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                    x = torch.cat([cls.expand(-1,N,-1), x], dim=0) # [50, 46, 768]
                    x = self._call_transformer_block(resblock, x) # [50, 46, 768]

                    if block_idx == self.last_layer+1:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        if self.use_siglip:
                            x = vit.post_layernorm(x[:, 0, :]) # [46, 768]
                            # SigLIP doesn't have a projection layer in the same way
                            if hasattr(self.model, 'visual_projection'):
                                x = x @ self.model.visual_projection.weight.t()
                        else:
                            x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                            if self.model.visual.proj is not None:
                                x = x @ self.model.visual.proj # [46, 512]
                        return x
                else:
                    x = self._call_transformer_block(resblock, x)

        elif fusion_mode == 'attn_masking':
            attn_mask = self.make_attn_mask(pred_masks)
            transformer_blocks = vit.encoder.layers if self.use_siglip else vit.transformer.resblocks
            for block_idx, resblock in enumerate(transformer_blocks):
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        N = pred_masks.shape[0]
                        x = x.expand(-1, N, -1)

                    x = self._call_transformer_block(resblock, x, attn_mask=attn_mask)

                    if block_idx == self.last_layer:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        if self.use_siglip:
                            x = vit.post_layernorm(x[:, 0, :]) # [46, 768]
                            if hasattr(self.model, 'visual_projection'):
                                x = x @ self.model.visual_projection.weight.t()
                        else:
                            x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                            if self.model.visual.proj is not None:
                                x = x @ self.model.visual.proj # [46, 512]
                        return x
                else:
                    x = self._call_transformer_block(resblock, x)

        elif fusion_mode == 'L2G':
            attn_mask = self.make_attn_mask(pred_masks)
            x1_x2 = torch.cat([x,x2], dim=1)
            transformer_blocks = vit.encoder.layers if self.use_siglip else vit.transformer.resblocks
            for block_idx, resblock in enumerate(transformer_blocks): 
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        x = x1_x2[:,:len(pred_masks),:]
                        x2 = x1_x2[:,len(pred_masks):,:]
                    x_ori_local = x.clone()
                    x = resblock(x)
                    x2 = self._call_transformer_block(resblock, x_ori_local+x2*2, attn_mask=attn_mask) 
                else:                    
                    x1_x2 = resblock(x1_x2)
            
                if block_idx == self.last_layer+1:
                    x2 = x2.permute(1, 0, 2) # [46, 50, 768]
                    if self.use_siglip:
                        x2 = vit.post_layernorm(x2[:, 0, :]) # [46, 768]
                        if hasattr(self.model, 'visual_projection'):
                            x2 = x2 @ self.model.visual_projection.weight.t()
                    else:
                        x2 = self.model.visual.ln_post(x2[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x2 = x2 @ self.model.visual.proj # [46, 512]
                    return x2

        elif fusion_mode == 'G2L':
            if self.use_siglip:
                pass
            #     attn_mask = self.make_attn_mask(pred_masks, use_siglip=True)
            #     x1_x2 = torch.cat([x,x2], dim=1)
            #     print(x1_x2.shape, 'x1_x2')
                
            #     transformer_blocks = vit.encoder.layers
            #     for block_idx, resblock in enumerate(transformer_blocks): 
            #         if block_idx >= masking_block:
            #             if block_idx == masking_block:
            #                 x = x1_x2[:,:len(pred_masks),:]
            #                 x2 = x1_x2[:,len(pred_masks):,:]
            #             x_ori_global = x2.clone()
                        
            #             x_ori_global = x_ori_global.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]

            #             x_ori_global = x_ori_global.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]

            #             x_ori_global = torch.mul(x_ori_global, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
            #             N = x_ori_global.size(0)
            #             x_ori_global = x_ori_global.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]
                        
            #             x_ori_global = x_ori_global.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                        
            #             x = self._call_transformer_block(resblock, x_ori_global*2+x) 
                        
            #             x2 = x2.permute(1,0,2)
            #             x2 = resblock(x2, attention_mask=None) 

            #             x = x.permute(1,0,2)
            #             x2 = x2.permute(1,0,2) # (HW, B, D)
            #         else:
            #             # print(x.shape)      # before fusion
            #             # print(x2.shape)
            #             # print(x1_x2.shape) # after fusion
            #             # print(torch.cuda.max_memory_allocated() / 1024**3)
            #             x1_x2 = resblock(x1_x2, attention_mask=None)
                        
                    
            #         if block_idx == self.last_layer+1:
            #             x = x.permute(1, 0, 2) # 276, 576, 768
            #             x = vit.post_layernorm(x) # 276, 576, 768
            #             x = vit.head(x) # 276, 768
                    
            #             return x
            else:
                attn_mask = self.make_attn_mask(pred_masks)
                # print(attn_mask.shape, pred_masks.shape, x2.shape)
                # torch.Size([3312, 197, 197]) torch.Size([276, 14, 14]) torch.Size([197, 276, 768])

                x1_x2 = torch.cat([x,x2], dim=1)
                transformer_blocks = vit.transformer.resblocks
                for block_idx, resblock in enumerate(transformer_blocks): 
                    if block_idx >= masking_block:
                        if block_idx == masking_block:
                            x = x1_x2[:,:len(pred_masks),:]
                            x2 = x1_x2[:,len(pred_masks):,:]
                        x_ori_global = x2.clone()
                        cls = x_ori_global[:1,:,:]
                        x_ori_global = x_ori_global[1:,:,:]
                        x_ori_global = x_ori_global.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]

                        x_ori_global = x_ori_global.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]

                        x_ori_global = torch.mul(x_ori_global, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                        N = x_ori_global.size(0)
                        x_ori_global = x_ori_global.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]

                        x_ori_global = x_ori_global.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                        x_ori_global = torch.cat([cls.expand(-1,N,-1), x_ori_global], dim=0) # [50, 46, 768]

                        x = self._call_transformer_block(resblock, x_ori_global*2+x) 
                        x2 = resblock(x2, attn_mask=attn_mask)
                    else:
                        x1_x2 = resblock(x1_x2)
                
                    if block_idx == self.last_layer+1:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                
                        x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x = x @ self.model.visual.proj # [46, 512]
                        return x

        elif fusion_mode == 'G2L&L2G':
            attn_mask = self.make_attn_mask(pred_masks)
            x1_x2 = torch.cat([x,x2], dim=1)
            # local
            transformer_blocks = vit.encoder.layers if self.use_siglip else vit.transformer.resblocks
            for block_idx, resblock in enumerate(transformer_blocks): 

                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        x = x1_x2[:,:len(pred_masks),:]
                        x2 = x1_x2[:,len(pred_masks):,:]
                        x_hybrid_local = x.clone()
                        x_hybrid_global = x2.clone()

                    x_ori_local = x.clone()
                    x_ori_global = x2.clone()
                    # token masking
                    cls = x_ori_global[:1,:,:]
                    x_ori_global = x_ori_global[1:,:,:]
                    x_ori_global = x_ori_global.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]
                    x_ori_global = x_ori_global.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]
                    x_ori_global = torch.mul(x_ori_global, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                    N = x_ori_global.size(0)
                    x_ori_global = x_ori_global.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]
                    x_ori_global = x_ori_global.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                    x_ori_global = torch.cat([cls.expand(-1,N,-1), x_ori_global], dim=0) # [50, 46, 768]

                    x = self._call_transformer_block(resblock, x)
                    x2 = self._call_transformer_block(resblock, x2, attn_mask=attn_mask)
                    x_hybrid_local = self._call_transformer_block(resblock, x_hybrid_local+2*x_ori_global)
                    x_hybrid_global = self._call_transformer_block(resblock, x_ori_local+2*x_hybrid_global, attn_mask=attn_mask)
                    
                else:
                    x1_x2 = resblock(x1_x2)
            
                if block_idx == self.last_layer+1:
                    x_hybrid_local = x_hybrid_local.permute(1, 0, 2) # [46, 50, 768]
                    if self.use_siglip:
                        x_hybrid_local = vit.post_layernorm(x_hybrid_local[:, 0, :]) # [46, 768]
                        if hasattr(self.model, 'visual_projection'):
                            x_hybrid_local = x_hybrid_local @ self.model.visual_projection.weight.t()
                    else:
                        x_hybrid_local = self.model.visual.ln_post(x_hybrid_local[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x_hybrid_local = x_hybrid_local @ self.model.visual.proj # [46, 512]

                    x_hybrid_global = x_hybrid_global.permute(1, 0, 2) # [46, 50, 768]
                    if self.use_siglip:
                        x_hybrid_global = vit.post_layernorm(x_hybrid_global[:, 0, :]) # [46, 768]
                        if hasattr(self.model, 'visual_projection'):
                            x_hybrid_global = x_hybrid_global @ self.model.visual_projection.weight.t()
                    else:
                        x_hybrid_global = self.model.visual.ln_post(x_hybrid_global[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x_hybrid_global = x_hybrid_global @ self.model.visual.proj # [46, 512]
                    return x_hybrid_local + x_hybrid_global


        return x




# === From backbone_desktop2.py ===

def get_activation(outputs, mode='full', is_train=False):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if is_train:
        if mode=='avg':
            def hook(model, input, output):
                if len(output.shape)==4: #CNN layers
                    outputs.append(output.mean(dim=[2,3]).clone())
                elif len(output.shape)==3: #ViT
                    if output.shape[1] == 1:
                        outputs.append(output[:, 0].clone())
                    elif output.shape[1] == 0:
                        outputs.append(output[0, :].clone())
                elif len(output.shape)==2: #FC layers
                    outputs.append(output.clone())
                    
        elif mode=='max':
            def hook(model, input, output):
                if len(output.shape)==4: #CNN layers
                    outputs.append(output.amax(dim=[2,3]).clone())
                elif len(output.shape)==3: #ViT
                    outputs.append(output[:, 0].clone())
                elif len(output.shape)==2: #FC layers
                    outputs.append(output.clone())
                    
        elif mode=='full':
            def hook(model, input, output):
                outputs.append(output.clone())
    else:
        if mode=='avg':
            def hook(model, input, output):
                if len(output.shape)==4: #CNN layers
                    outputs.append(output.mean(dim=[2,3]).detach().cpu())
                elif len(output.shape)==3: #ViT
                    if output.shape[1] == 1:
                        outputs.append(output[:, 0].detach().cpu())
                    elif output.shape[1] == 0:
                        outputs.append(output[0, :].detach().cpu())
                elif len(output.shape)==2: #FC layers
                    outputs.append(output.detach().cpu())
                    
        elif mode=='max':
            def hook(model, input, output):
                if len(output.shape)==4: #CNN layers
                    outputs.append(output.amax(dim=[2,3]).detach().cpu())
                elif len(output.shape)==3: #ViT
                    outputs.append(output[:, 0].detach().cpu())
                elif len(output.shape)==2: #FC layers
                    outputs.append(output.detach().cpu())
                    
        elif mode=='full':
            def hook(model, input, output):
                outputs.append(output.detach())
    return hook


class clip_backbone(nn.Module):
    ''' CLIP backbone before attention pooling.'''

    def __init__(self, model_name = 'RN50', visual_projs_path = './pretrain/'):
        '''
        Args:
            model_name: availabe models = ['RN50', 'RN101', 'RN50x4', 'RN50x64']
            visual_projs_path = path to 'clip_weight.pth'
        '''
        super().__init__()

        self.model, _ = clip.load(model_name)

        self.visual_projs_path = visual_projs_path + model_name + '_clip_weights.pth'

        self.in_channels = self.model.visual._inplanes # visual feature dimension
        self.text_channels = self.model.visual.output_dim # text feature dimension

        self.v_proj = nn.Conv2d(self.in_channels, self.in_channels, 1).to(self.device)
        self.c_proj = nn.Conv2d(self.in_channels, self.text_channels, 1).to(self.device)

        v_proj_weight = nn.parameter.Parameter(self.model.visual.attnpool.v_proj.weight[:,:,None,None])
        c_proj_weight = nn.parameter.Parameter(self.model.visual.attnpool.c_proj.weight[:,:,None,None])

        self.v_proj.weight = v_proj_weight
        self.c_proj.weight = c_proj_weight

        # self.load_visual_projs()

        self.activation_map = None
        self.activation_map_gradients = None

        # self.text_encoder_attn_mask = self.model.build_attention_mask()

    def load_visual_projs(self):
        print('load visual projs')
        loaded = torch.load(self.visual_projs_path, map_location='cuda')

        for attr in ['v_proj', 'c_proj']:
            current_attr = getattr(self, attr) # self.v_proj
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:,:,None,None]
            current_attr.load_state_dict(state_dict)

    @property
    def device(self):
        return self.model.visual.conv1.weight.device

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype


    def forward(self, data, text):
        image = data['img'][0] # [1,3,480,480]
        H,W,_ = data['img_metas'][0][0]['ori_shape']


        x = self.model.encode_image(image.type(self.dtype))
        h, w = x.shape[2], x.shape[3]
        v = self.v_proj(x)

        image_features = self.c_proj(v) # [4, 1024, 15, 15]
        text_features = self.model.encode_text(text).unsqueeze(-1) # [4, 1024, 1]


        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True) # [4(batch), 1024(channel), 15(H), 15(W)]
        text_features = text_features / text_features.norm(dim=1, keepdim=True) # [4(batch), 1024(channel), 1]

        # # upsaple clip's feature map into original image shape
        # image_features = F.interpolate(image_features, size=(H,W), align_corners=False, mode='bilinear')


        ## for score map
        image_features = rearrange(image_features, 'b c h w -> b (h w) c')


        score_map = rearrange(torch.einsum('bij,bjk->bki', image_features, text_features), 'b s (h w) -> b s h w', h=h, w=w)
        # score_map = score_map / score_map.norm(dim=1, keepdim=True)

        # Upsample
        score_map = F.interpolate(score_map, size=(H,W), align_corners=False, mode='bilinear')

        image_features = rearrange(image_features, 'b (h w) c -> b c h w', h=h, w=w)

        return score_map, image_features, text_features

    def get_image_feature(self, data, free_solo=True, size=None):

        # For Free_SOLO(Detectron2) Input
        if free_solo:
            image = data.to(self.device)

            if size:
                self.H, self.W = size
            # self.H, self.W = image.shape[2], image.shape[3] # shorter size = 800

            x = self.model.encode_image(image.type(self.dtype))
            v = self.v_proj(x)

            image_features = self.c_proj(v)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            return image_features

        # For MMdetection Input
        image = data['img'][0] # [1,3,H,W]
        self.H, self.W,_ = data['img_metas'][0][0]['ori_shape']

        x = self.model.encode_image(image.type(self.dtype))
        v = self.v_proj(x)

        image_features = self.c_proj(v) # [batch, text_channel, H, W]

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features

    def get_text_feature(self, text, target_noun_index=None):
        text_features = self.model.encode_text(text, target_noun_index) #[batch, channel]

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    def text_masking_feature(self, text, masking_index=[], masking_block=11):
        text_encoder = self.model.transformer
        masking_index = [i+1 for i in masking_index] # because start token

        # sentence = text[0]
        # noun_phrase = text[1]


        # text = [1, 77]
        x = self.model.token_embedding(text).type(self.dtype) # [1,77,512]
        x = x + self.model.positional_embedding.type(self.dtype) # [1,77,512]
        x = x.permute(1, 0, 2) # [77, 1, 512]

        for block_idx, resblock in enumerate(text_encoder.resblocks): # last block idx [11, 11, 23]
            if block_idx >= masking_block:
                if masking_index:
                    x[masking_index] = 0
                    x = resblock(x)

                else:
                    x = resblock(x)
            else:
                x = resblock(x)


        x = x.permute(1, 0, 2) # [1, 77, 512]
        x = self.model.ln_final(x).type(self.dtype) # [1, 77, 512]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection # [1, 512]
        x = x / x.norm(dim=1, keepdim=True)

        return x



    def get_ensembled_text_feature(self, text):
        # text = [80, 77]
        text_features = self.model.encode_text(text) # text_features = [80,1024]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features.mean(0)
        text_features = text_features / text_features.norm() # text_features = [1024] -> [1,1024,1]로 바꿔줘야 한다.

        return text_features[None,:,None] # [1,1024,1]

    def get_score_map(self, image_features, text_features):
        h, w = image_features.shape[2], image_features.shape[3]

        image_features = rearrange(image_features, 'b c h w -> b (h w) c')
        score_map = rearrange(torch.einsum('bij,bjk->bki', image_features, text_features), 'b s (h w) -> b s h w', h=h, w=w)

        # Upsample
        score_map = F.interpolate(score_map, size=(self.H,self.W), align_corners=False, mode='bilinear')

        return score_map

    def get_gloval_vector(self, data, attn_mask=None, free_solo=True):
        image = data.to(self.device)
        if free_solo:
            image_features = self.model.encode_image(image.type(self.dtype), attn=True, attn_mask=attn_mask)

            # normalize
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            return image_features # [1, 1024]

    def project_seg_attn(self, data, pred_masks):
        image, pred_masks = data.to(self.device), pred_masks.to(self.device)

        image_features = self.model.encode_image(image.type(self.dtype), attn=False) # [1,2048,7,7]
        image_features = image_features / image_features.norm(dim=1, keepdim=True) # normalize

        H, W = image_features.shape[2], image_features.shape[3]

        pred_masks = TF.resize(pred_masks, (H,W)) # [N,7,7]

        mask_features = []

        mean_position = torch.ones(1, dtype=torch.bool).to(self.device)
        for pred_mask in pred_masks:
            # attn_mask should be [HW+1, HW+1] and dtype=torch.bool
            attn_mask = pred_mask.reshape(pred_mask.shape[0] * pred_mask.shape[1]) # [49]
            attn_mask = torch.cat([mean_position, attn_mask],dim=0).repeat(pred_mask.shape[0] * pred_mask.shape[1]+1,1) # [50,50]

            attn_mask_feature = self.model.visual.attnpool(image_features, attn_mask=~attn_mask)
            # attn_mask_feature = attn_mask_feature / attn_mask_feature.norm(dim=1, keepdim=True)

            mask_features.append(attn_mask_feature)

        mask_features = torch.stack(mask_features, dim=0) # [N,1,1024]

        return mask_features

    def generate_score_map(self, image, text_features, H,W):
        image_features = self.model.encode_image(image.type(self.dtype), attn=False) # [1,2048,7,7]
        image_features = self.v_proj(image_features) # [1, 2048, 7, 7]
        image_features = self.c_proj(image_features) # [1, 1024, 7, 7]

        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        h, w = image_features.shape[2], image_features.shape[3]

        image_features = rearrange(image_features, 'b c h w -> b (h w) c') # [1,49,1024]
        # text features = [1, 1024]
        score_map = torch.einsum('bij,bjk->bki', image_features, text_features[:,:,None]) # [1,1,49]
        score_map = rearrange(score_map, 'b s (h w) -> b s h w', h=h, w=w) # [1, 1, 7, 7]
        score_map = F.interpolate(score_map, size=(H,W), align_corners=False, mode='bilinear') # [1, 1, 224, 224]
        score_map = (score_map - torch.min(score_map)) / (torch.max(score_map) - torch.min(score_map))

        return score_map


    def feature_map_masking(self, data, pred_masks, query_masking=True, ignore_zero=False,
                            before_normalize=True, after_normalize=False, parallel=True, use_attn_mask=False):
        image, pred_masks = data.to(self.device), pred_masks.to(self.device)

        image_features = self.model.encode_image(image.type(self.dtype), attn=False) # [1,2048,7,7]
        if before_normalize:
            image_features = image_features / image_features.norm(dim=1, keepdim=True) # normalize

        H, W = image_features.shape[2], image_features.shape[3]

        # pred_masks = TF.resize(pred_masks, (H,W)) # [N,7,7]
        try:
            pred_masks = TF.resize(pred_masks.type(torch.float32), (H, W))  # [N,7,7]
        except:
            pred_masks = pred_masks.cpu()
            pred_masks = TF.resize(pred_masks.type(torch.float32), (H, W))  # [N,7,7]
            pred_masks = pred_masks.to(self.device)


        if parallel:
            masked_feature_map = torch.mul(image_features, pred_masks[:, None, :, :])

            if query_masking:
                masked_features = self.model.visual.attnpool(masked_feature_map, ignore_zero=ignore_zero)
            else:
                masked_features = self.model.visual.attnpool(masked_feature_map, image_features)

            if after_normalize:
                masked_features = masked_features / masked_features.norm(dim=1, keepdim=True)


        else:
            masked_features = []
            mean_position = torch.ones(1, dtype=torch.bool).to(self.device)
            # masking feature map
            for pred_mask in pred_masks: # [N,7,7], [1,2048,7,7]
                masked_feature_map = torch.mul(image_features, pred_mask[None, None, ...])

                if use_attn_mask:
                    attn_mask = pred_mask.reshape(pred_mask.shape[0] * pred_mask.shape[1])  # [49]
                    attn_mask = torch.cat([mean_position, attn_mask], dim=0).repeat(pred_mask.shape[0] * pred_mask.shape[1] + 1, 1)  # [50,50]
                    masked_feature = self.model.visual.attnpool(masked_feature_map, ignore_zero=ignore_zero, attn_mask=~attn_mask)
                else:
                    masked_feature = self.model.visual.attnpool(masked_feature_map, ignore_zero=ignore_zero)

                if after_normalize:
                    masked_feature = masked_feature / masked_feature.norm(dim=1, keepdim=True)

                masked_features.append(masked_feature)

            masked_features = torch.stack(masked_features, dim=0)

        return masked_features

    def calculate_similarity_score(self, image_features, text_features, ours=False):
        # image_features = [N, 1,1024]
        # text_feature = [1,1024]
        # logit_scale.exp() = 100. ?? 이거는 학습에 사용되는 건데 inference시에도 곱해줄 필요가 있을까?

        # text_features = text_features.squeeze(-1)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()

        # logits_per_image = logit_scale * image_features @ text_features.t() # 16.5868

        logits_per_image = logit_scale * image_features @ text_features.t() # 0.1659
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        if ours:
            logits_per_image = torch.mean(logits_per_image, 1)
        return logits_per_image # [N, 1, 1]


    def generate_grad_cam(self, image, noun_text, H=224, W=224, Tokenize=True):
        # noun_text에 대한 tokenize 이후 embedding을 입력 받아야 한다.
        # noun_text에 a photo of a 붙여주기
        image, noun_text = image.to(self.device), noun_text.to(self.device)

        if Tokenize:
            noun_text_feature = self.model.encode_text(noun_text) # [batch, channel]
            noun_text_feature = noun_text_feature / noun_text_feature.norm(dim=1, keepdim=True)
        else:
            noun_text_feature = noun_text

        # Get global vector and image_feature_map
        image_feature_map = self.model.encode_image(image.type(self.dtype), attn=False)
        image_feature_map.register_hook(self.save_gradients)
        global_vector = self.model.visual.attnpool(image_feature_map)
        global_vector = global_vector / global_vector.norm(dim=1, keepdim=True) # this is for backward function

        image_feature_map = image_feature_map / image_feature_map.norm(dim=1, keepdim=True) # For Grad-CAM and CAM

        # Get gradients of feature map
        logit_scale = self.model.logit_scale.exp()
        logit_per_image = logit_scale * global_vector @ noun_text_feature.t()
        logit_per_image *= -1

        self.zero_grad()
        logit_per_image.backward(retain_graph=True)

        # Get Grad-CAM
        gradients_map = self.activation_map_gradients
        weights = torch.mean(gradients_map, axis=(2,3))
        weighted_activations = weights[:,:,None,None] * image_feature_map
        Grad_CAM = weighted_activations.sum(axis=1)


        Grad_CAM = F.interpolate(Grad_CAM[None,:,:,:], size=(H,W), align_corners=False, mode='bilinear')


        return Grad_CAM

    def save_gradients(self, grad):
        self.activation_map_gradients = grad

    def save_activation_map(self, input):
        self.activation_map = input

    def get_activation_map(self):
        return self.activation_map

    def get_gradients(self):
        return self.activation_map_gradients


class clip_vit(nn.Module):
    def __init__(self, model_name='ViT-B/16'):
        super().__init__()
        '''
        model_neam: 'RN50', 'RN101', 'RN50x4', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT0L/14@336px'
        '''


        self.model, _ = clip.load(model_name)

    @property
    def device(self):
        return self.model.visual.conv1.weight.device

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype

    def get_image_feature(self, data):
        image = data.to(self.device)

        image_feature = self.model.encode_image(image) # [1, 50, 512]

        # normalize
        image_feature = image_feature / image_feature.norm(dim=2, keepdim=True)

        image_feature = image_feature[:, 1:, :]
        h, w = int(np.sqrt(image_feature.shape[1])), int(np.sqrt(image_feature.shape[1]))

        image_feature = image_feature.reshape(image_feature.shape[0], h, w, image_feature.shape[2])
        image_feature = image_feature.permute(0,3,1,2)

        return image_feature



    def get_text_feature(self, text):
        text_features = self.model.encode_text(text).unsqueeze(-1) #[batch, channel, 1]

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features



    def get_score_map(self, image_features, text_features):
        h, w = image_features.shape[2], image_features.shape[3]

        image_features = rearrange(image_features, 'b c h w -> b (h w) c')

        score_map = rearrange(torch.einsum('bij,bjk->bki', image_features, text_features), 'b s (h w) -> b s h w', h=h, w=w)

        # # Upsample
        # score_map = F.interpolate(score_map, size=(self.h,self.w), align_corners=False, mode='bilinear')

        return score_map

    def befor_masking(self, data, pred_masks, pixel_mean):
        image = data.to(self.device)

        h, w = image.shape[2], image.shape[3]
        pred_masks = TF.resize(pred_masks, (h, w)).type(torch.float32)

        image_features = []

        masked_img = []
        for pred_mask in pred_masks:
            img = image * pred_mask[None, None, ...] + (1 - pred_mask[None, None, ...]) * pixel_mean
            masked_img.append(img)
        masked_img = torch.stack(masked_img, dim=0).squeeze(1)



        image_features = self.get_image_feature(masked_img).squeeze(0)

        return image_features

    def after_masking(self, data, pred_masks):
        image_features = self.forward(data)

        H, W = image_features.shape[2], image_features.shape[3]

        pred_masks = TF.resize(pred_masks, (H,W)) # [N,7,7]

        for pred_mask in pred_masks:
            masked_feature_map = torch.mul(image_features, pred_mask[None, None, ...])
            mean_feature = masked_feature_map.mean()


        return image_features

class CLIPViTFM_Desktop(nn.Module):
    def __init__(self, model_name='ViT-B/32', size=224, dfn=False, blip=False, siglip=False):
        super().__init__()

        if model_name == 'ViT-B/32':
            self.last_layer = 11
            self.num_heads = 12
        elif model_name == 'ViT-B/16':
            self.last_layer = 11
            self.num_heads = 12
        elif model_name == 'ViT-L/14':
            self.last_layer = 23
            self.num_heads = 16
        elif model_name == 'ViT-H/14':
            self.last_layer = 31
            self.num_heads = 16
        elif model_name == 'ViT-SO400M-14':
            self.last_layer = 26
            self.num_heads = 16

        self.dfn = dfn
        self.blip = blip
        self.siglip = siglip
        
        if self.dfn:
            # self.model, _ = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-dfn')
            self.model, _ = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14')
            # print(self.model.context_length)
        elif self.siglip:
            self.model, _ = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP') #ViT-SO400M-14-SigLIP-384')
            print(self.model)
            # from transformers import AutoModel, AutoProcessor
            # model_id = "google/siglip-base-patch16-384"
            # self.model = AutoModel.from_pretrained(model_id)

        elif self.blip:
            self.model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        else:
            self.model, _ = clip.load(model_name)

        # import inspect
        # target = self.model.encode_image  # bound method
        # func = target.__func__            # actual function object
        
        # print("Function:", func.__name__)
        # print("Defined at:", inspect.getsourcefile(func))
        # print("Line number:", inspect.getsourcelines(func)[1])

    @property
    def device(self):
        return self.model.visual.conv1.weight.device

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype

    def encode_text_attention(self, noun_phrase_token, sentence_token, fusion_layer=10):
        text_encoder = self.model.transformer
        
        # Step 1: Embed inputs
        np_embed = self.model.token_embedding(noun_phrase_token).type(self.dtype) + self.model.positional_embedding.type(self.dtype)
        sent_embed = self.model.token_embedding(sentence_token).type(self.dtype) + self.model.positional_embedding.type(self.dtype)
    
        # [1, 77, 512] → [77, 1, 512]
        np_embed = np_embed.permute(1, 0, 2)
        sent_embed = sent_embed.permute(1, 0, 2)
    
        # Step 2: Process through transformer layers
        for block_idx, resblock in enumerate(text_encoder.resblocks):
            if block_idx < fusion_layer:
                np_embed = resblock(np_embed)
                sent_embed = resblock(sent_embed)
    
            elif block_idx == fusion_layer:
                # Step 3: Fuse the self-attention outputs
                np_ln = resblock.ln_1(np_embed) # [77, 1, 512]
                sent_ln = resblock.ln_1(sent_embed)
    
                np_attn = resblock.attn(np_ln, np_ln, np_ln)[0] # [77, 1, 512]
                sent_attn = resblock.attn(sent_ln, sent_ln, sent_ln)[0]
    
                # Fuse attention embeddings (e.g., average or concat + proj)
                fused_attn = 0.5 * np_attn + 0.5 * sent_attn
    
                # Add residual
                fused_embed = 0.5 * (np_embed + sent_embed) + fused_attn
    
                # MLP part
                fused_embed = fused_embed + resblock.mlp(resblock.ln_2(fused_embed)) # [77, 1, 512]
    
            else:
                # Use fused representation for the remaining layers
                fused_embed = resblock(fused_embed)
    
        # Step 4: Final projection
        fused_embed = fused_embed.permute(1, 0, 2)  # [1, seq_len, dim]
        fused_embed = self.model.ln_final(fused_embed).type(self.dtype)
    
        # Step 5: Extract CLS or EOS token
        final_embedding = fused_embed[torch.arange(fused_embed.shape[0]), noun_phrase_token.argmax(dim=-1)]
        return final_embedding @ self.model.text_projection

    def text_masking_feature(self, text, masking_index=[], masking_block=10):
        text_encoder = self.model.transformer
        masking_index = [i+1 for i in masking_index] # because start token


        x = self.model.token_embedding(text).type(self.dtype) # [1,77,512]
        x = x + self.model.positional_embedding.type(self.dtype) # [1,77,512]
        x = x.permute(1, 0, 2) # [77, 1, 512]

        for block_idx, resblock in enumerate(text_encoder.resblocks): # last block idx [11, 11, 23]
            if block_idx >= masking_block:
                if masking_index:
                    x[masking_index] = 0
                    x = resblock(x)

                else:
                    x = resblock(x)
            else:
                x = resblock(x)


        x = x.permute(1, 0, 2) # [1, 77, 512]
        x = self.model.ln_final(x).type(self.dtype) # [1, 77, 512]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection # [1, 512]

        return x

    def calculate_score(self, image_features, text_features, visual_norm_dim=1, ours=False):
        # image_features = [N,512]
        # text_feature = [1,512]
        # logit_scale.exp() = 100.

        image_features = image_features / image_features.norm(dim=visual_norm_dim, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        if self.blip:
            logit_scale = 1.
        else:
            logit_scale = self.model.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t() # 0.1659
        if ours:
            logits_per_image = torch.mean(logits_per_image, 1)
        return logits_per_image # [N, 1]

    def upsample_pos_emb(self, emb):
        # upsample the pretrained embedding for higher resolution
        # emb size NxD
        first = emb[:1, :]
        emb = emb[1:, :]
        N, D = emb.size(0), emb.size(1)
        n = int(np.sqrt(N))
        assert n * n == N

        emb = emb.permute(1, 0)
        emb = emb.view(1, D, n, n).contiguous()
        emb = F.upsample(emb, size=(self.size), mode='bilinear',
                         align_corners=None)
        emb = emb.view(D, -1).contiguous()
        emb = emb.permute(1, 0)
        emb = torch.cat([first, emb], 0)
        emb = nn.parameter.Parameter(emb.half())
        return emb

    def make_attn_mask(self, pred_masks, size=None, use_siglip=False):
        # pred_masks = [46, H, W], Torch.bool
        if size is not None:
            pred_masks = TF.resize(pred_masks, size=(size, size))  # [46,7,7]

        if not use_siglip:

            cls = torch.ones((1,), dtype=torch.bool).to(self.device)
    
            N = pred_masks.size(0)
            attn_masks = pred_masks.view(N, -1) # [46, 49]
            attn_masks = [attn_mask.expand(self.num_heads, -1) for attn_mask in attn_masks] # [N, num_heads, L]
            attn_masks = torch.stack(attn_masks, dim=0).view(N * self.num_heads, -1).contiguous() # [N*num_heads, L]
            attn_masks = torch.cat([cls.expand(N * self.num_heads, -1), attn_masks], dim=1) # [N*num_heads, L+1]
            attn_masks = attn_masks.unsqueeze(-1).expand(-1, -1, attn_masks.shape[1]) # [N*num_heads, L+1, L+1]

            return ~attn_masks
        else:
            return pred_masks*1.0


    # def make_attn_mask(self, pred_masks, size=None):
    #     # pred_masks =  [46, H, W], Torch.bool
    #     if size is not None:
    #         pred_masks = TF.resize(pred_masks, size=(size, size))  # [46,7,7]
    #     N, H, W = pred_masks.size()
    #     attn_masks = torch.ones((N * self.num_heads, H*W+1, H*W+1), dtype=torch.bool).to(self.device)
    #     attn_masks[:, 0, 1:] = pred_masks[:,None,:,:].expand(-1,self.num_heads,-1,-1).reshape(N * self.num_heads, -1)
    #     return ~attn_masks
    
    def get_cvpr_vector(self, local_imgs, global_imgs, pred_masks, masking_block=None, fusion_mode='G2L'):
        torch.cuda.empty_cache()
        
        if masking_block is None:
            masking_block = self.last_layer

        vit = self.model.visual

        x = local_imgs.type(self.model.dtype)
        

        if fusion_mode == 'crop': # [1, 512]
            x = vit(x)
            return x[:, 0, :]
        x = vit.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        ####################### x2 ####################################
        if global_imgs is not None:
            x2 = global_imgs.type(self.model.dtype) #.to(x.device)
            x2 = vit.conv1(x2)  # shape = [*, width, grid, grid]
            x2 = x2.reshape(x2.shape[0], x2.shape[1], -1)
            x2 = x2.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x2 = torch.cat([vit.class_embedding.to(x2.dtype) + torch.zeros(x2.shape[0], 1, x2.shape[-1],
                                                                        dtype=x2.dtype, device=x2.device), x2], dim=1)  # shape = [*, grid ** 2 + 1, width]

            x2 = x2 + vit.positional_embedding.to(x2.dtype)
            x2 = vit.ln_pre(x2)

            x2 = x2.permute(1, 0, 2)  # NLD -> LND

        L, N, D = x[1:, :, :].size(0), x.size(1), x.size(2)
        size = int(np.sqrt(L))
        assert size * size == L

        pred_masks = TF.resize(pred_masks.type(torch.float32), (size, size))
        if fusion_mode == 'token_masking':
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx >= masking_block:
                    cls = x[:1,:,:]
                    x = x[1:,:,:]
                    x = x.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]

                    x = x.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]

                    x = torch.mul(x, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                    N = x.size(0)
                    x = x.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]

                    x = x.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                    x = torch.cat([cls.expand(-1,N,-1), x], dim=0) # [50, 46, 768]
                    x = resblock(x) # [50, 46, 768]

                    if block_idx == self.last_layer:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x = x @ self.model.visual.proj # [46, 512]
                        return x
                else:
                    x = resblock(x)

        elif fusion_mode == 'attn_masking':
            attn_mask = self.make_attn_mask(pred_masks)
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        N = pred_masks.shape[0]
                        x = x.expand(-1, N, -1)

                    x = resblock(x, attn_mask=attn_mask)

                    if block_idx == self.last_layer:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x = x @ self.model.visual.proj # [46, 512]
                        return x
                else:
                    x = resblock(x)

        elif fusion_mode == 'L2G':
            attn_mask = self.make_attn_mask(pred_masks)
            x1_x2 = torch.cat([x,x2], dim=1)
            for block_idx, resblock in enumerate(vit.transformer.resblocks): 
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        x = x1_x2[:,:len(pred_masks),:]
                        x2 = x1_x2[:,len(pred_masks):,:]
                    x_ori_local = x.clone()
                    x = resblock(x)
                    x2 = resblock(x_ori_local+x2*2,attn_mask=attn_mask) 
                else:                    
                    x1_x2 = resblock(x1_x2)
            
                if block_idx == self.last_layer:
                    x2 = x2.permute(1, 0, 2) # [46, 50, 768]
                    x2 = self.model.visual.ln_post(x2[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        x2 = x2 @ self.model.visual.proj # [46, 512] 
                    return x2

        elif fusion_mode == 'G2L':
            attn_mask = self.make_attn_mask(pred_masks)
            x1_x2 = torch.cat([x,x2], dim=1)
            for block_idx, resblock in enumerate(vit.transformer.resblocks): 
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        x = x1_x2[:,:len(pred_masks),:]
                        x2 = x1_x2[:,len(pred_masks):,:]
                    x_ori_global = x2.clone()
                    cls = x_ori_global[:1,:,:]
                    x_ori_global = x_ori_global[1:,:,:]
                    x_ori_global = x_ori_global.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]

                    x_ori_global = x_ori_global.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]

                    x_ori_global = torch.mul(x_ori_global, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                    N = x_ori_global.size(0)
                    x_ori_global = x_ori_global.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]

                    x_ori_global = x_ori_global.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                    x_ori_global = torch.cat([cls.expand(-1,N,-1), x_ori_global], dim=0) # [50, 46, 768]

                    x = resblock(x_ori_global * 2 + x) # (7*7+1, num_masks, 768)
                    x2 = resblock(x2, attn_mask)
                else:
                    x1_x2 = resblock(x1_x2)
                    
                if block_idx == self.last_layer:
                    del x1_x2
                    x = x.permute(1, 0, 2) # [46, 50, 768]
                    x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        x = x @ self.model.visual.proj # [46, 512] 

                    return x

        elif fusion_mode == 'G2L&L2G':
            attn_mask = self.make_attn_mask(pred_masks)
            x1_x2 = torch.cat([x,x2], dim=1)
            # local
            for block_idx, resblock in enumerate(vit.transformer.resblocks): 

                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        x = x1_x2[:,:len(pred_masks),:]
                        x2 = x1_x2[:,len(pred_masks):,:]
                        x_hybrid_local = x.clone()
                        x_hybrid_global = x2.clone()

                    x_ori_local = x.clone()
                    x_ori_global = x2.clone()
                    # token masking
                    cls = x_ori_global[:1,:,:]
                    x_ori_global = x_ori_global[1:,:,:]
                    x_ori_global = x_ori_global.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]
                    x_ori_global = x_ori_global.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]
                    x_ori_global = torch.mul(x_ori_global, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                    N = x_ori_global.size(0)
                    x_ori_global = x_ori_global.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]
                    x_ori_global = x_ori_global.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                    x_ori_global = torch.cat([cls.expand(-1,N,-1), x_ori_global], dim=0) # [50, 46, 768]

                    x = resblock(x)
                    x2 = resblock(x2,attn_mask=attn_mask)
                    x_hybrid_local = resblock(x_hybrid_local+2*x_ori_global) 
                    x_hybrid_global = resblock(x_ori_local+2*x_hybrid_global, attn_mask=attn_mask)
                    
                else:
                    x1_x2 = resblock(x1_x2)
            
                if block_idx == self.last_layer:
                    x_hybrid_local = x_hybrid_local.permute(1, 0, 2) # [46, 50, 768]
                    x_hybrid_local = self.model.visual.ln_post(x_hybrid_local[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        x_hybrid_local = x_hybrid_local @ self.model.visual.proj # [46, 512] 

                    x_hybrid_global = x_hybrid_global.permute(1, 0, 2) # [46, 50, 768]
                    x_hybrid_global = self.model.visual.ln_post(x_hybrid_global[:, 0, :]) # [46, 768]
                    if self.model.visual.proj is not None:
                        x_hybrid_global = x_hybrid_global @ self.model.visual.proj # [46, 512] 
                    return x_hybrid_local + x_hybrid_global


        return x



    def forward(self, image, pred_masks, masking_block=None, masking_type='token_masking', target_layers=['visual.transformer.resblocks[31].ls_2'], pool_mode='full'): # [f'visual.trunk.blocks[25].ls2']
        # if self.dfn:
            
        #     if masking_type not in ['ours_local', 'crop']:
        #         try:
        #             pred_masks = TF.resize(pred_masks.type(torch.float32), (image.shape[-2], image.shape[-1]))  # [N,7,7]
        #         except:
        #             pred_masks = pred_masks.cpu()
        #             pred_masks = TF.resize(pred_masks.type(torch.float32), (image.shape[-2], image.shape[-1]))  # [N,7,7]
        #             pred_masks = pred_masks.to(self.device)

        #         image = image * pred_masks.unsqueeze(1)
        #         x = self.model.encode_image(image)
        #     elif masking_type in ['crop']:
        #         x = self.model.encode_image(image)
        #     else:
        #         all_features = {target_layer:[] for target_layer in target_layers}
        #         hooks = {}
        #         for target_layer in target_layers:
        #             command = "self.model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        #             hooks[target_layer] = eval(command)
        #         with torch.no_grad():
        #             image_features = self.model.encode_image(image)
        #             orig_inter_features = all_features[target_layers[0]][0]
        #             print(self.model.visual)
        #             inter_features = self.model.visual.ln_post(- orig_inter_features[:, 0, :]) 
        #             new_features = inter_features @ self.model.visual.proj
        #             new_features /= new_features.norm(dim=-1, keepdim=True)
        #             new_features = new_features[1:, :]
            
        # else:
        if masking_block is None:
            masking_block = self.last_layer

        if self.blip:
            vit = self.model.vision_model
        else:
            vit = self.model.visual
        try:
            x = image.type(self.model.dtype)
        except:
            x = image
    
        if masking_type == 'crop': # [1, 512]
            if self.blip:
                x = vit(x)
                return x[0][:, 0, :]
            elif self.siglip:
                outputs = vit(x)
                return outputs
            else:
                x = vit(x)
                return x[:, :] # 0, :]

        if not self.siglip:
            if self.blip:
                x = vit.embeddings.patch_embedding(x)
            # elif self.siglip:
            #     x = vit.trunk.patch_embed(x)
            #     x = vit.trunk.patch_drop(x)
            #     x = vit.trunk.norm_pre(x)
            else:
                x = vit.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        else:
            x2 = image.type(next(self.model.visual.parameters()).dtype)

            attn_mask = self.make_attn_mask(pred_masks, size=224, use_siglip=True)
            global_outputs = vit(x2*attn_mask.unsqueeze(1)) 

            if masking_type != 'ours_local':
                return global_outputs

        # size = x.shape[2], x.shape[3]
        if not self.siglip:
            x = x.reshape(x.shape[0], x.shape[1], -1)
    
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        if not self.blip and not self.siglip:
            x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + vit.positional_embedding.to(x.dtype)
            # x = x + self.original_pos_embedding
            x = vit.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND

            L, N, D = x[1:, :, :].size(0), x.size(1), x.size(2)

            size = int(np.sqrt(L))
            assert size * size == L
        else:
            N, L, D = x[:, :, :].size(0), x.size(1), x.size(2)

            size = int(np.sqrt(L))
            assert size * size == L
            

        if masking_type not in ['ours_local', 'ours_local_wo_negation']:
            try:
                pred_masks = TF.resize(pred_masks.type(torch.float32), (size, size))  # [N,7,7]
            except:
                pred_masks = pred_masks.cpu()
                pred_masks = TF.resize(pred_masks.type(torch.float32), (size, size))  # [N,7,7]
                pred_masks = pred_masks.to(self.device)

        if masking_type == 'token_masking' or masking_type == 'ours_global':
            if not self.blip and not self.siglip:
                for block_idx, resblock in enumerate(vit.transformer.resblocks):
                    if block_idx >= masking_block:
                        cls = x[:1,:,:]
                        x = x[1:,:,:]
                        x = x.permute(1,2,0) # [49, 1, 768] -> [1, 768, 49]
    
                        x = x.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]
    
                        x = torch.mul(x, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                        N = x.size(0)
                        x = x.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]
    
                        x = x.permute(2,0,1) # [46, 768, 49] -> [49, 46, 768]
                        x = torch.cat([cls.expand(-1,N,-1), x], dim=0) # [50, 46, 768]
                        x = resblock(x) # [50, 46, 768]
                        
                        if block_idx == self.last_layer:
                            x = x.permute(1, 0, 2) # [46, 50, 768]
                            if masking_type == 'ours_global': 
                                x = self.model.visual.ln_post(-x[:, 1:, :]) # [46, 49, 768]
                            else:
                                x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                            if self.model.visual.proj is not None:
                                x = x @ self.model.visual.proj # [46, 512] or [46, 49, 512]
                            if masking_type == 'ours_global': 
                                x = x / x.norm(dim=-1, keepdim=True)
                                # indices = torch.arange(1, x.shape[1]+1, dtype=x.dtype)
                                # weights = torch.log1p(indices) / torch.log1p(indices).sum() #(indices / indices.sum())
                                # weights = weights.to(x.device)
                                # x = x * weights[None,:,None]
                                # x = torch.mean(x, 1)
    
                    else:
                        x = resblock(x)
                        
            elif self.blip:
                for block_idx, resblock in enumerate(vit.encoder.layers):
                    if block_idx >= masking_block:
                        x = x[:,:,:]
                        x = x.permute(0, 2, 1) # [49, 1, 768] -> [1, 768, 49]
                        
                        x = x.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]
                    
                        x = torch.mul(x, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                        N = x.size(0)
                        x = x.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]
                     
                        x = x.permute(0, 2, 1)
                        # image_atts = torch.ones(x.size()[:-1], dtype=torch.long).to(x.device)
                        x = resblock(x, attention_mask=None, output_attentions=False)[0] # [50, 46, 768]
                        
                        if block_idx == self.last_layer:
                            if masking_type == 'ours_global': 
                                x = self.model.vision_model.post_layernorm(-x[:, 1:, :]) # [46, 49, 768]
                            else:
                                x = self.model.vision_model.post_layernorm(x[:, 0, :]) # [46, 768]
                            x = x @ self.model.vision_proj.weight # [46, 512] or [46, 49, 512]
                                
                            if masking_type == 'ours_global': 
                                x = x / x.norm(dim=-1, keepdim=True)
                                # indices = torch.arange(1, x.shape[1]+1, dtype=x.dtype)
                                # weights = torch.log1p(indices) / torch.log1p(indices).sum() #(indices / indices.sum())
                                # weights = weights.to(x.device)
                                # x = x * weights[None,:,None]
                                # x = torch.mean(x, 1)
    
                    else:
                        # image_atts = torch.ones(x.size()[:-1], dtype=torch.long).to(x.device)
                        x = resblock(x, attention_mask=None, output_attentions=False)[0]

            elif self.siglip:
                for block_idx, block in enumerate(vit.trunk.blocks):
                
                    if block_idx >= masking_block:
                        # ==========================================
                        # 1) Reshape tokens to spatial (no CLS)
                        # ==========================================
                        # (N, L, D) -> (B, D, S, S)
                        x = x[:,:,:]
                        x = x.permute(0, 2, 1) # [49, 1, 768] -> [1, 768, 49]
                        
                        x = x.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]
                                        
                        # ==========================================
                        # 2) Apply spatial mask
                        # pred_masks: [B, S, S]
                        # ==========================================
                        x = x * pred_masks[:, None, :, :]
                
                        # ==========================================
                        # 3) Back to tokens
                        # ==========================================
                        N = x.size(0)
                        x = x.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]
                     
                        x = x.permute(0, 2, 1)
                
                        # ==========================================
                        # 4) Apply transformer block
                        # ==========================================
                        x = block(x)
                    
                    # ==========================================
                    # 5) LAST BLOCK OUTPUT
                    # ==========================================
                    if block_idx == self.last_layer:
                        x = vit.trunk.norm(x)  

                x = vit.trunk.forward_head(x)
                x = vit.head(x)
                return x
                                
                

        elif masking_type == 'ours_local': 
            # print(self.blip, 'blip')
            if masking_block == 'avg':
                layer_outputs = []
                for block_idx, resblock in enumerate(vit.transformer.resblocks):
                    x = resblock(x) # [50, 46, 768]
                    if block_idx >= 6: #0: # 6
                        y = x.clone()
                        y = y.permute(1, 0, 2) # [46, 50, 768]
                        y = self.model.visual.ln_post(-y[:, 1:, :]) # [46, 49, 768]
                        if self.model.visual.proj is not None:
                            y = y @ self.model.visual.proj # [46, 49, 512]
                        layer_outputs.append(y.clone())

                x = torch.stack(layer_outputs, dim=0).mean(dim=0) 
                # x = torch.mean(x, 1)
            else:
                if self.blip:
                    for block_idx, resblock in enumerate(vit.encoder.layers):
                        if block_idx <= masking_block:
                            x = resblock(x, attention_mask=None, output_attentions=False)[0]
        
                            if block_idx == masking_block:
                                x = x.permute(1, 0, 2) # [46, 50, 768]
                                x = self.model.vision_model.post_layernorm(-x[:, 1:, :]) # [46, 49, 768]
                                x = x @ self.model.vision_proj.weight # [46, 49, 512]

                elif self.siglip:                        
                    for block_idx, resblock in enumerate(vit.trunk.blocks):
                        if block_idx <= masking_block:
                            x = resblock(x)

                            if block_idx == masking_block:
                                x = vit.trunk.norm(-x[:, :, :])

                elif masking_block == 'dynamic':
                    masking_layer = range(20, 32)
                    max_layer = max(masking_layer)
                    
                    layer_features = []
                    
                    for block_idx, resblock in enumerate(vit.transformer.resblocks):
                    
                        # stop after highest masking layer
                        if block_idx > max_layer:
                            break
                    
                        x = resblock(x)  # forward progressively
                    
                        if block_idx in masking_layer:
                            tmp = x.permute(1, 0, 2)              # [46, 50, 768]
                            tmp = self.model.visual.ln_post(-tmp[:, 1:, :])  # remove CLS → [46, 49, 768]
                    
                            if self.model.visual.proj is not None:
                                tmp = tmp @ self.model.visual.proj  # [46, 49, 512]
                    
                            layer_features.append(tmp)
                    
                    # stack
                    x = torch.stack(layer_features, dim=0)

                elif masking_block == 'dynamic_s':
                    masking_layer = range(6, 12)
                    max_layer = max(masking_layer)
                    
                    layer_features = []
                    
                    for block_idx, resblock in enumerate(vit.transformer.resblocks):
                    
                        # stop after highest masking layer
                        if block_idx > max_layer:
                            break
                    
                        x = resblock(x)  # forward progressively
                    
                        if block_idx in masking_layer:
                            tmp = x.permute(1, 0, 2)              # [46, 50, 768]
                            tmp = self.model.visual.ln_post(-tmp[:, 1:, :])  # remove CLS → [46, 49, 768]
                    
                            if self.model.visual.proj is not None:
                                tmp = tmp @ self.model.visual.proj  # [46, 49, 512]
                    
                            layer_features.append(tmp)
                    
                    # stack
                    x = torch.stack(layer_features, dim=0)

                elif masking_block == 'dynamic_s_32':
                    masking_layer = range(6, 12)
                    max_layer = max(masking_layer)
                    
                    layer_features = []
                    
                    for block_idx, resblock in enumerate(vit.transformer.resblocks):
                    
                        # stop after highest masking layer
                        if block_idx > max_layer:
                            break
                    
                        x = resblock(x)  # forward progressively
                    
                        if block_idx in masking_layer:
                            tmp = x.permute(1, 0, 2)              # [46, 50, 768]
                            tmp = self.model.visual.ln_post(-tmp[:, 1:, :])  # remove CLS → [46, 49, 768]
                    
                            if self.model.visual.proj is not None:
                                tmp = tmp @ self.model.visual.proj  # [46, 49, 512]
                    
                            layer_features.append(tmp)
                    
                    # stack
                    x = torch.stack(layer_features, dim=0)
                       
                else:
                    for block_idx, resblock in enumerate(vit.transformer.resblocks):
                        if block_idx <= masking_block:
                            x = resblock(x) # [50, 46, 768]
        
                            if block_idx == masking_block:
                                x = x.permute(1, 0, 2) # [46, 50, 768]
                                x = self.model.visual.ln_post(-x[:, 1:, :]) # [46, 49, 768]
                                if self.model.visual.proj is not None:
                                    x = x @ self.model.visual.proj # [46, 49, 512]
                                    # indices = torch.arange(1, x.shape[1]+1, dtype=x.dtype)
                                    # weights = torch.log1p(indices) / torch.log1p(indices).sum() #(indices / indices.sum())
                                    # weights = weights.to(x.device)
                                    # x = x * weights[None,:,None]
                                    # x = torch.mean(x, 1)

        elif masking_type == 'ours_local2': 
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx <= masking_block:
                    x = resblock(x) # [50, 46, 768]

                    if block_idx == masking_block:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        x = self.model.visual.ln_post(-x[:, 1:, :]) # [46, 49, 768]
                        x = x.permute(0, 2, 1) # [46, 768, 49]
                        N = x.size(0)
                        x = x.view(N, D, size, size).contiguous() # [1, 768, 49] -> [1, 768, 7, 7]

                        x = torch.mul(x, pred_masks[:, None, :, :]) # [46, 768, 7, 7]
                        N = x.size(0)
                        x = x.view(N, D, L).contiguous() # [46, 768, 7, 7] -> [46, 768, 49]

                        x = x.permute(0, 2, 1)
                        
                        if self.model.visual.proj is not None:
                            x = x @ self.model.visual.proj # [46, 49, 512]
                            x = torch.mean(x, 1)

        elif masking_type == 'ours_local_wo_negation': 
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx <= masking_block:
                    x = resblock(x) # [50, 46, 768]

                    if block_idx == masking_block:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        x = self.model.visual.ln_post(x[:, 1:, :]) # [46, 49, 768]
                        if self.model.visual.proj is not None:
                            x = x @ self.model.visual.proj # [46, 49, 512]

        elif masking_type == 'specific_masking':
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx == masking_block:
                    cls = x[:1, :, :]
                    x = x[1:, :, :]
                    x = x.permute(1, 2, 0)  # [49, 1, 768] -> [1, 768, 49]

                    x = x.view(N, D, size, size).contiguous()  # [1, 768, 49] -> [1, 768, 7, 7]

                    x = torch.mul(x, pred_masks[:, None, :, :])  # [46, 768, 7, 7]
                    N = x.size(0)
                    x = x.view(N, D, L).contiguous()  # [46, 768, 7, 7] -> [46, 768, 49]

                    x = x.permute(2, 0, 1)  # [46, 768, 49] -> [49, 46, 768]
                    x = torch.cat([cls.expand(-1, N, -1), x], dim=0)  # [50, 46, 768]
                    x = resblock(x)  # [50, 46, 768]
                else:
                    x = resblock(x)

                if block_idx == self.last_layer:
                    x = x.permute(1, 0, 2) # [46, 50, 768]
                    x = self.model.visual.ln_post(x[:,0,:])
                    if self.model.visual.proj is not None:
                        x = x @ self.model.visual.proj

        elif masking_type == 'attn_masking':
            attn_mask = self.make_attn_mask(pred_masks)
            for block_idx, resblock in enumerate(vit.transformer.resblocks):
                if block_idx >= masking_block:
                    if block_idx == masking_block:
                        N = pred_masks.shape[0]
                        x = x.expand(-1, N, -1)

                    x = resblock(x, attn_mask=attn_mask)

                    if block_idx == self.last_layer:
                        x = x.permute(1, 0, 2) # [46, 50, 768]
                        x = self.model.visual.ln_post(x[:, 0, :]) # [46, 768]
                        if self.model.visual.proj is not None:
                            x = x @ self.model.visual.proj # [46, 512]

                else:
                    x = resblock(x)

        return x

        



class CLIPMaskedSpatialViT(nn.Module):
    def __init__(self, model_name='ViT-B/32', upsample=1, start_block=0, align_corners=None):
        super(CLIPMaskedSpatialViT, self).__init__()

        if model_name == 'ViT-B/32':
            self.target_size = 7
            self.patch_size = 32
        elif model_name == 'ViT-B/16':
            self.target_size = 14
            self.patch_size = 16
        elif model_name == 'ViT-L/14':
            self.target_size = 16
            self.patch_size = 14

        self.model, self.preprocess = clip.load(model_name)

        self.align_corners = align_corners

        assert (upsample == 1) or (upsample & (upsample-1) == 0)  # power of 2
        self.upsample = upsample
        self.target_size = self.target_size * self.upsample
        self.stem_stride = self.patch_size // upsample
        self.model.visual.conv1.stride = self.stem_stride
        self.model.visual.conv1.padding = (
            self.patch_size - 1) // 2  # TODO: make it more precise
        self.model.visual.positional_embedding = self.upsample_pos_emb(
            self.model.visual.positional_embedding)

        self.start_block = start_block

    @property
    def device(self):
        return self.model.visual.conv1.weight.device

    @property
    def dtype(self):
        return self.model.visual.conv1.weight.dtype

    def upsample_pos_emb(self, emb):
        # upsample the pretrained embedding for higher resolution
        # emb size NxD
        first = emb[:1, :]
        emb = emb[1:, :]
        N, D = emb.size(0), emb.size(1)
        size = int(np.sqrt(N))
        assert size * size == N
        new_size = size * self.upsample
        emb = emb.permute(1, 0)
        emb = emb.view(1, D, size, size).contiguous()
        emb = F.upsample(emb, size=new_size, mode='bilinear',
                         align_corners=self.align_corners)
        emb = emb.view(D, -1).contiguous()
        emb = emb.permute(1, 0)
        emb = torch.cat([first, emb], 0)
        emb = nn.parameter.Parameter(emb.half())
        return emb

    def masks_to_attn_map(self, masks):
        # masks size NxHxW
        N = masks.size(0)
        # masks is 1 for the object and 0 for others, need to invert it
        masks = 1 - masks.bool().float()
        # interpolate to target size
        masks = masks.unsqueeze(1).float()
        target_size = (self.target_size, self.target_size)
        masks = F.interpolate(masks, size=target_size,
                              mode='nearest', align_corners=None)
        masks = masks.squeeze(1)
        attn_map = masks.view(N, -1)
        attn_map = torch.cat([attn_map, 1-torch.eye(N).to(attn_map.device)], 1)
        attn_map = attn_map.bool().half() * (-100)
        return attn_map

    def encode_text(self, text):
        return self.model.encode_text(text)
        

    def encode_image(self, image):
        return self.model.encode_image(image)

    def forward(self, im, masks):
        vit = self.model.visual
        x = im.type(self.model.dtype)

        x = vit.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        attn_mask = self.masks_to_attn_map(masks)
        attn_mask = attn_mask.type(self.model.dtype)
        num_masks = attn_mask.size(0)
        for block_idx, resblock in enumerate(vit.transformer.resblocks):
            if block_idx == self.start_block:
                gv = x[:1]
                gv = gv.repeat(num_masks, 1, 1)  # LND
            if block_idx >= self.start_block:
                attn = resblock.attn
                source = resblock.ln_1(torch.cat([x[1:], gv], 0))
                gv = gv + attn(
                    source[-num_masks:],
                    source,
                    source,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
                gv = gv + resblock.mlp(resblock.ln_2(gv))
            x = resblock(x)

        gv = gv.permute(1, 0, 2)
        gv = vit.ln_post(gv)
        if vit.proj is not None:
            gv = (gv.view(-1, gv.size(-1)) @
                  vit.proj).view(gv.size(0), gv.size(1), -1)

        return gv # image_features

    def get_mask_feature(self, image, pred_masks):
        with torch.no_grad():
            image = image.to(self.device)
            image_features = self.forward(image, pred_masks) # [1,46,512]

            # image_features = image_features.permute(1,2,0)

            image_features = image_features / image_features.norm(dim=2, keepdim=True)

        return image_features # [46,512,1]


    def get_text_feature(self, text):
        text_features = self.model.encode_text(text).unsqueeze(-1) #[batch, channel, 1]

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    def calculate_similarity_score(self, image_features, text_features):
        # image_features = [N, 1,1024]
        # text_feature = [1,1024,1]
        # logit_scale.exp() = 100. ?? 이거는 학습에 사용되는 건데 inference시에도 곱해줄 필요가 있을까?

        # image_features = [N, 512, 1]
        # text_features = [1, 512, 1]
        # image_features = image_features.squeeze(2)

        text_features = text_features.squeeze(-1)



        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()

        # logits_per_image = logit_scale * image_features @ text_features.t() # 16.5868

        logits_per_image = logit_scale * image_features @ text_features.t() # 0.1659
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image.squeeze(0) # [1, N, 1]