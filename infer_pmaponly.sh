#!/bin/bash

python inference_newmask2former_vitb16_pmaponly.py
# python inference_newmask2former_vitb16_pmaponly.py --split test --splitBy umd
# python inference_newmask2former_vitb16_pmaponly.py --splitBy google


# python inference_newmask2former_vitb16_pmaponly.py --dataset refcoco --splitBy unc 
# python inference_newmask2former_vitb16_pmaponly.py --dataset refcoco --split testA --splitBy unc 
# python inference_newmask2former_vitb16_pmaponly.py --dataset refcoco --split testB --splitBy unc 

# python inference_newmask2former_vitb16_pmaponly.py --dataset refcoco+ --splitBy unc
# python inference_newmask2former_vitb16_pmaponly.py --dataset refcoco+ --split testA --splitBy unc 
# python inference_newmask2former_vitb16_pmaponly.py --dataset refcoco+ --split testB --splitBy unc 