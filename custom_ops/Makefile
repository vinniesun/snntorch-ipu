CXX ?= g++
CXXFLAGS = -std=c++14 -fPIC -g
LDLIBS = -shared -lpopart
ONNX_NAMESPACE = -DONNX_NAMESPACE=onnx

BUILD_DIR = ../so_file
SOURCE1 = ./heaviside_custom_op.cpp
SOURCE2 = ./straight_through_estimator.cpp
SOURCE3 = ./fast_sigmoid.cpp
SOURCE4 = ./sigmoid.cpp
TARGET1 = $(BUILD_DIR)/heaviside_custom_ops.so
TARGET2 = $(BUILD_DIR)/straight_through_estimator_custom_ops.so
TARGET3 = $(BUILD_DIR)/fast_sigmoid_custom_ops.so

all: create_build_dir heaviside straight_through_estimator fast_sigmoid

.PHONY: create_build_dir
create_build_dir: 
	mkdir -p $(BUILD_DIR)

heaviside: ./heaviside_custom_op.cpp 
	$(CXX) $(SOURCE1)  $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(TARGET1)

straight_through_estimator: ./straight_through_estimator.cpp
	$(CXX) $(SOURCE2)  $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(TARGET2)

fast_sigmoid: ./fast_sigmoid.cpp
	$(CXX) $(SOURCE3)  $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(TARGET3)

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
