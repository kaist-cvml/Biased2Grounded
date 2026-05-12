#!/bin/bash
source /home/nsml/.venv/bin/activate

# python multilingual_inference_siglip2_top3_final_textlayer.py
# python multilingual_inference_siglip2_top3_final_textlayer.py --split test --splitBy umd
# python multilingual_inference_siglip2_top3_final_textlayer.py --splitBy google


python multilingual_inference_siglip2_top3_final_textlayer.py --dataset refcoco --splitBy unc 
# python multilingual_inference_siglip2_top3_final_textlayer.py --dataset refcoco --split testA --splitBy unc 
# python multilingual_inference_siglip2_top3_final_textlayer.py --dataset refcoco --split testB --splitBy unc 

# python multilingual_inference_siglip2_top3_final_textlayer.py --dataset refcoco+ --splitBy unc
# python multilingual_inference_siglip2_top3_final_textlayer.py --dataset refcoco+ --split testA --splitBy unc 
# python multilingual_inference_siglip2_top3_final_textlayer.py --dataset refcoco+ --split testB --splitBy unc 