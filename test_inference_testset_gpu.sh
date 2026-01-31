#!/bin/bash -v
python test_inference_testset_v5.py output_test_4ch_gpu/UK6L1WZQ6/model_00000 \
    --config config/bathrooms_test_config_4ch_gpu.yaml \
    --json-dir data/bathroom_2.2k_filter \
    --num-scenes 5 \
    --split train \
    --verbose

