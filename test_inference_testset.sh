#!/bin/bash -v
python test_inference_testset_v4.py output_test_4ch/EA7AZJE4X/model_00000 \
    --config config/bathrooms_test_config_4ch.yaml \
    --json-dir data/bathroom_2.2k_filter \
    --num-scenes 20 \
    --split train \
    --verbose

