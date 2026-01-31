#!/bin/bash -v
python test_inference.py output_test/WI3NACUD1/model_00001 \
    --config config/bathrooms_test_config.yaml \
    --output generated_bathroom.png \
    --max-boxes 10
