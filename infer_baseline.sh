#!/bin/bash

python inference_newmask2former_vitb16_baseline.py
# python inference_newmask2former_vitb16_baseline.py --split test --splitBy umd
# python inference_newmask2former_vitb16_baseline.py --splitBy google


# python inference_newmask2former_vitb16_baseline.py --dataset refcoco --splitBy unc 
# python inference_newmask2former_vitb16_baseline.py --dataset refcoco --split testA --splitBy unc 
# python inference_newmask2former_vitb16_baseline.py --dataset refcoco --split testB --splitBy unc 

# python inference_newmask2former_vitb16_baseline.py --dataset refcoco+ --splitBy unc
# python inference_newmask2former_vitb16_baseline.py --dataset refcoco+ --split testA --splitBy unc 
# python inference_newmask2former_vitb16_baseline.py --dataset refcoco+ --split testB --splitBy unc 