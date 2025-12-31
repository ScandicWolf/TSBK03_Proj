// Basic includes for OpenGL context via MicroGlut.
#include "MicroGlut.h"
#include "GL_utilities.h"

// Simple complex number used for GPU buffers (float2).
struct Complex
{
    float x;
    float y;
};

// Compile and link a compute shader from file path (used by ocean module).
GLuint loadComputeShader(const char *path);

// Compute 2D inverse FFT: takes spectrum SSBO (row-major kx fastest), runs column then row inverse passes.
// Returns SSBO with time-domain data (un-normalized, divide by W*H to get original amplitudes).
GLuint computeIFFT2D(GLuint spectrumSSBO, int W, int H);
