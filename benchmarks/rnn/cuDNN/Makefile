BENCH_NAME ?= lstm_cell_cudnn
BUILD_DIR  := build
OUTPUT_FILE ?= ../c_cudnn_lstm_cell_bench.tsv

.PHONY: build bench clean

build:
	@mkdir -p build && cd build && cmake .. && make -j

$(BUILD_DIR)/$(BENCH_NAME): build

bench: $(BUILD_DIR)/$(BENCH_NAME)
	@./$(BUILD_DIR)/$(BENCH_NAME) $(OUTPUT_FILE)

clean:
	@rm -rf build
