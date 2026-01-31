#!/bin/bash -v
python test_inference_testset_v5.py output_test_4ch/0M3W2MIJN/model_00007 \
    --config config/bathrooms_test_config_4ch.yaml \
    --json-dir data/bathroom_2.2k_filter \
    --num-scenes 5 \
    --split train \
    --verbose

