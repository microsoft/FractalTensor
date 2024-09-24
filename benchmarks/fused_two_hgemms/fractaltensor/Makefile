BENCH_NAME ?= back2back_hgemm
BUILD_DIR  := build

.PHONY: build clean

build:
	@mkdir -p build && cd build && cmake .. && make -j12

$(BUILD_DIR)/$(BENCH_NAME): build

clean:
	@rm -rf build
