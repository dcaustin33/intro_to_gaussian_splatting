# CUDA paths
CUDA_PATH       := /usr/local/cuda
LIBTORCH_PATH   := /path/to/libtorch

# Architecture
ARCH            := sm_75

# Compiler and flags
NVCC            := $(CUDA_PATH)/bin/nvcc
GPP             := g++
CXXFLAGS        := -std=c++17 -O2 -I$(CUDA_PATH)/include -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
LDFLAGS         := -L$(CUDA_PATH)/lib64 -lcudart -L$(LIBTORCH_PATH)/lib -ltorch -ltorch_python -lc10

# Source files
SRCS            := my_program.cpp render_engine.cu
OBJS            := render_engine.o

# Targets
all: my_program

render_engine.o: render_engine.cu
	$(NVCC) -c render_engine.cu -o render_engine.o -arch=$(ARCH)

my_program: $(OBJS) my_program.cpp
	$(GPP) my_program.cpp $(OBJS) -o my_program $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f *.o my_program