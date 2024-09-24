find kaleido/ -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i
find benchmarks/ -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i
