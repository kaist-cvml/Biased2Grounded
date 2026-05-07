#!/bin/bash
source /home/nsml/.venv/bin/activate

# python multilingual_inference_siglip2_top3_centroid_textlayer_cost.py
# python multilingual_inference_siglip2_top3_centroid_textlayer_cost.py --split test --splitBy umd
# python multilingual_inference_siglip2_top3_centroid_textlayer_cost.py --splitBy google


python multilingual_inference_siglip2_top3_centroid_textlayer_cost.py --dataset refcoco --splitBy unc 
# python multilingual_inference_siglip2_top3_centroid_textlayer_cost.py --dataset refcoco --split testA --splitBy unc 
# python multilingual_inference_siglip2_top3_centroid_textlayer_cost.py --dataset refcoco --split testB --splitBy unc 

# python multilingual_inference_siglip2_top3_centroid_textlayer_cost.py --dataset refcoco+ --splitBy unc
# python multilingual_inference_siglip2_top3_centroid_textlayer_cost.py --dataset refcoco+ --split testA --splitBy unc 
# python multilingual_inference_siglip2_top3_centroid_textlayer_cost.py --dataset refcoco+ --split testB --splitBy unc 