#Please set the following library root directories properly:
CNN_DIR=/dt/cnn/cnn
EIGEN_DIR=/dt/cnn/eigen

CNN_BUILD_DIR=${CNN_DIR}/build/cnn

all: classify
classify: classify.cc
	g++ -g -o  classify classify.cc  -I ${EIGEN_DIR} -I ${CNN_DIR} -std=c++11 -g -L/usr/lib  -lboost_program_options -lboost_serialization -L${CNN_BUILD_DIR} -lcnn 
