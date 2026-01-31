#!/bin/bash -v
#export KMP_DUPLICATE_LIB_OK=TRUE
#export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
#export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
#export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
python scripts/train_network.py config/bathrooms_test_config_4ch_gpu.yaml output_test_4ch_gpu/
