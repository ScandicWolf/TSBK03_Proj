# set this variable to the directory in which you saved the common files
commondir = ../common

SRC_DIR = src
INC_DIR = include

SOURCES = \
	main.cpp \
	$(SRC_DIR)/fft_gpu.cpp \
	$(SRC_DIR)/ocean_init.cpp \
	$(SRC_DIR)/ocean.cpp \
	$(SRC_DIR)/camera.cpp \
	$(SRC_DIR)/scene.cpp \
	$(SRC_DIR)/ocean_spectrum.cpp

CXX = g++
CXXFLAGS = -Wall -I$(commondir) -I$(INC_DIR) -I$(commondir)/Linux -DGL_GLEXT_PROTOTYPES
LDFLAGS = -lXt -lX11 -lGL -lm

all: main

main: $(SOURCES) $(commondir)/LoadTGA.c $(commondir)/GL_utilities.c $(commondir)/Linux/MicroGlut.c
	$(CXX) $(CXXFLAGS) -o $@ \
		$(commondir)/LoadTGA.c \
		$(commondir)/GL_utilities.c \
		$(commondir)/Linux/MicroGlut.c \
		$(SOURCES) \
		$(LDFLAGS)

clean:
	rm -f main

