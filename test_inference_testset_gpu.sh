#!/bin/bash -v
python test_inference_testset_v5.py output_test_4ch_gpu/0K1OTEWYX/model_00010 \
    --config config/bathrooms_test_config_4ch_gpu.yaml \
    --json-dir data/bathroom_2.2k_filter \
    --num-scenes 5 \
    --verbose \
    --split train
    #--split test

