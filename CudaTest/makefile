run:
	nvcc kernel.cu -O3 -lcublas -lcurand -lcudart -std=c++20 -o test2.exe
debug:
	nvcc kernel.cu -g -G -O0 -std=c++17 -o test.exe