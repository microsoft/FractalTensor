BENCH_NAME ?= stacked_lstm
BUILD_DIR  := build
OUTPUT_FILE ?= ../../cute_stacked_lstm_bench.tsv

.PHONY: build bench clean

build:
	@mkdir -p build && cd build && cmake .. && make -j

$(BUILD_DIR)/$(BENCH_NAME): build

bench: $(BUILD_DIR)/$(BENCH_NAME)
	@./$(BUILD_DIR)/$(BENCH_NAME) $(OUTPUT_FILE)

clean:
	@rm -rf build
