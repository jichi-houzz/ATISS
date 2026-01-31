#!/bin/bash -v
python test_inference_testset_v5.py output_test_4ch/MSS6CN1OJ/model_00150 \
    --config config/bathrooms_test_config_4ch.yaml \
    --json-dir data/bathroom_2.2k_filter \
    --num-scenes 20 \
    --split test \
    --verbose

