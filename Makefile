#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

CUDNN_HOME ?=
BUILD_DIR := build
BENCHMARK_DIR := benchmarks
BENCHMARK_MODEL_CLASS ?= rnn
BENCHMARK_PROJ ?= fractaltensor
BENCHMARK_MODEL ?= stacked_lstm

.PHONY: build clean install-python test-backend test-frontend benchmark benchmarks cpp-format

build:
	@mkdir -p $(BUILD_DIR)/
	@cd build && cmake ../ -D PYTHON_EXECUTABLE:FILEPATH=`which python3` \
	-D CUDNN_INCLUDE_DIR=$(CUDNN_HOME)/include \
	-D CUDNN_LIBRARY=$(CUDNN_HOME)/lib/libcudnn.so && make -j$(nproc)

$(BUILD_DIR)/kaleido: 
	@$(MAKE) build

install-python:
	@pip install -r requirements.txt

test-frontend:
	@./scripts/tests/frontend_unit_tests.sh
	@./scripts/tests/frontend_examples.sh

test-backend: $(BUILD_DIR)/kaleido
	@./scripts/tests/backend_unit_tests.sh

benchmark: $(BUILD_DIR)/kaleido
	@cd $(BENCHMARK_DIR)/$(BENCHMARK_MODEL_CLASS)/$(BENCHMARK_PROJ)/$(BENCHMARK_MODEL) && \
	mkdir -p build && cd build && cmake .. && make -j$(nproc)

benchmarks:
	@./scripts/benchmarks/bench.sh

cpp-format:
	@./scripts/cpp_format.sh

clean:
	@rm -f unittest.log
	@rm -rf $(BUILD_DIR)
