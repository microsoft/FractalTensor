#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

CUDNN_HOME ?=
BUILD_DIR := build

.PHONY: build clean

build:
	@mkdir -p $(BUILD_DIR)/
	@cd build && cmake ../ -D PYTHON_EXECUTABLE:FILEPATH=`which python3` \
	-D CUDNN_INCLUDE_DIR=$(CUDNN_HOME)/include \
	-D CUDNN_LIBRARY=$(CUDNN_HOME)/lib/libcudnn.so && make -j$(nproc)

$(BUILD_DIR)/kaleido: 
	@$(MAKE) build

clean:
	@rm -f unittest.log
	@rm -rf $(BUILD_DIR)
