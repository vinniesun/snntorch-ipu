CXX ?= g++
CXXFLAGS = -std=c++14 -fPIC -g
LDLIBS = -shared -lpopart
ONNX_NAMESPACE = -DONNX_NAMESPACE=onnx

BUILD_DIR = build
SOURCE1 = ./custom_ops/heaviside_custom_op.cpp
SOURCE2 = ./custom_ops/straight_through_estimator.cpp
TARGET1 = $(BUILD_DIR)/heaviside_custom_ops.so
TARGET2 = $(BUILD_DIR)/straight_through_estimator_custom_ops.so

all: create_build_dir heaviside straight_through_estimator

.PHONY: create_build_dir
create_build_dir: 
	mkdir -p $(BUILD_DIR)

heaviside: ./custom_ops/heaviside_custom_op.cpp 
	$(CXX) $(SOURCE1)  $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(TARGET1)

straight_through_estimator: ./custom_ops/straight_through_estimator.cpp
	$(CXX) $(SOURCE2)  $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(TARGET2)

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
