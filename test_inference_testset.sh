#!/bin/bash -v
#python test_inference_testset_v2.py output_test/epoch50_N3BO74H4M/model_00049 \
#    --num-scenes 10 \
#    --verbose

#python test_inference_testset_v2.py output_test_4ch/jvvdluvdm/model_00001 \
#    --config config/bathrooms_test_config_4ch.yaml \
#    --num-scenes 10 \
#    --verbose

#python test_inference_testset_v3.py output_test_4ch/epoch100_XRD075MG8/model_00100 \
#    --config config/bathrooms_test_config_4ch.yaml \
#    --json-dir data/bathroom_2.2k_filter \
#    --num-scenes 10 \
#    --split train \
#    --verbose

#python test_inference_testset_v3.py output_test_4ch/GP0ODX41W/model_00000 \
python test_inference_testset_v3.py output_test_4ch_gpu/QCPVUZBIE/model_00020 \
    --config config/bathrooms_test_config_4ch.yaml \
    --json-dir data/bathroom_2.2k_filter \
    --num-scenes 20 \
    --split train \
    --verbose

