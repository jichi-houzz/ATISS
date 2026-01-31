#!/bin/bash -v
python test_inference_testset.py output_test/epoch50_N3BO74H4M/model_00001 \
    --num-scenes 10 \
    --verbose
