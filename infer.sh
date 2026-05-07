#!/bin/bash

python inference_newmask2former_vitb16.py
# python inference_newmask2former_vitb16.py --split test --splitBy umd
# python inference_newmask2former_vitb16.py --splitBy google


# python inference_newmask2former_vitb16.py --dataset refcoco --splitBy unc 
# python inference_newmask2former_vitb16.py --dataset refcoco --split testA --splitBy unc 
# python inference_newmask2former_vitb16.py --dataset refcoco --split testB --splitBy unc 

# python inference_newmask2former_vitb16.py --dataset refcoco+ --splitBy unc
# python inference_newmask2former_vitb16.py --dataset refcoco+ --split testA --splitBy unc 
# python inference_newmask2former_vitb16.py --dataset refcoco+ --split testB --splitBy unc 