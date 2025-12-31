#include "fft_gpu.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static GLuint fft2DProgram = 0;


static const int kFFTLocalSize = 256; // Must match compute shader local_size_x

static GLint gFFT2DLocLength = -1;
static GLint gFFT2DLocStride = -1;
static GLint gFFT2DLocCount = -1;
static GLint gFFT2DLocStage = -1;
static GLint gFFT2DLocDir = -1;

static inline bool isPowerOfTwo(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

static inline int ilog2i(int value)
{
    int result = 0;
    while (value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

static inline int ceilDiv(int numerator, int denominator)
{
    return (numerator + denominator - 1) / denominator;
}


static bool cacheFFT2DUniforms()
{
    if (fft2DProgram == 0)
        return false;
    if (gFFT2DLocLength >= 0)
        return true;
    gFFT2DLocLength = glGetUniformLocation(fft2DProgram, "u_length");
    gFFT2DLocStride = glGetUniformLocation(fft2DProgram, "u_stride");
    gFFT2DLocCount = glGetUniformLocation(fft2DProgram, "u_count");
    gFFT2DLocStage = glGetUniformLocation(fft2DProgram, "u_stage");
    gFFT2DLocDir = glGetUniformLocation(fft2DProgram, "u_dir");
    if (gFFT2DLocLength < 0 || gFFT2DLocStride < 0 || gFFT2DLocCount < 0 || gFFT2DLocStage < 0 || gFFT2DLocDir < 0)
    {
        printf("FFT 2D compute shader missing uniforms.\n");
        return false;
    }
    return true;
}


static GLuint executeFFT2DPass(GLuint readBuffer, GLuint writeBuffer, int length, int stride, int count, int dir)
{
    if (length < 2 || count < 1)
        return readBuffer;

    glUniform1i(gFFT2DLocLength, length);
    glUniform1i(gFFT2DLocStride, stride);
    glUniform1i(gFFT2DLocCount, count);
    glUniform1i(gFFT2DLocDir, dir);

    glUniform1i(gFFT2DLocStage, -1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, readBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, writeBuffer);
    int groups = ceilDiv(length * count, kFFTLocalSize);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    GLuint src = writeBuffer;
    GLuint dst = readBuffer;
    const int log2Length = ilog2i(length);
    for (int stage = 1; stage <= log2Length; ++stage)
    {
        glUniform1i(gFFT2DLocStage, stage);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, src);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dst);
        int threads = (length / 2) * count;
        groups = ceilDiv(threads, kFFTLocalSize);
        glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        GLuint tmp = src;
        src = dst;
        dst = tmp;
    }
    return src;
}

static GLuint runFFT2D(GLuint primary, GLuint scratch, int W, int H, int dir)
{
    if (fft2DProgram == 0 || primary == 0 || scratch == 0 || W < 1 || H < 1)
        return 0;
    if (!isPowerOfTwo(W) || !isPowerOfTwo(H))
    {
        printf("FFT 2D dimensions must be powers of two (got %d x %d).\n", W, H);
        return 0;
    }
    if (!cacheFFT2DUniforms())
        return 0;

    glUseProgram(fft2DProgram);

    GLuint current = executeFFT2DPass(primary, scratch, W, 1, H, dir);
    if (current == 0)
        return 0;
    GLuint spare = (current == primary) ? scratch : primary;

    current = executeFFT2DPass(current, spare, H, W, W, dir);
    if (current == 0)
        return 0;

    return current;
}

GLuint loadComputeShader(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (!file)
    {
        printf("Could not open compute shader file: %s\n", path);
        return 0;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    char *src = (char *)malloc(size + 1);
    fread(src, 1, size, file);
    src[size] = '\0';
    fclose(file);
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, (const GLchar **)&src, NULL);
    glCompileShader(shader);
    free(src);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char info[512];
        glGetShaderInfoLog(shader, 512, NULL, info);
        printf("Compute shader compile error:\n%s\n", info);
        return 0;
    }
    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        char info[512];
        glGetProgramInfoLog(program, 512, NULL, info);
        printf("Compute shader link error:\n%s\n", info);
        return 0;
    }
    glDeleteShader(shader);
    return program;
}

static void initFFT2DProgram()
{
    if (fft2DProgram == 0)
    {
        fft2DProgram = loadComputeShader("shaders/fft_2d_stage.comp");
        if (fft2DProgram == 0)
        {
            printf("Failed to load 2D FFT compute shader.\n");
        }
    }
}

// Compute 2D inverse FFT: takes spectrum SSBO (row-major kx fastest), runs column then row inverse passes.
// Returns SSBO with time-domain data (un-normalized, divide by W*H to get original amplitudes).
GLuint computeIFFT2D(GLuint spectrumSSBO, int W, int H)
{
    if (spectrumSSBO == 0 || W < 1 || H < 1)
    {
        printf("computeIFFT2D: invalid arguments.\n");
        return 0;
    }
    initFFT2DProgram();
    if (fft2DProgram == 0)
        return 0;
    int size = W * H;
    GLuint ssboA = 0, ssboB = 0;
    glGenBuffers(1, &ssboA);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboA);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Complex) * size, NULL, GL_DYNAMIC_COPY);

    glBindBuffer(GL_COPY_READ_BUFFER, spectrumSSBO);
    glBindBuffer(GL_COPY_WRITE_BUFFER, ssboA);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizeof(Complex) * size);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

    glGenBuffers(1, &ssboB);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboB);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Complex) * size, NULL, GL_DYNAMIC_COPY);

    GLuint result = runFFT2D(ssboA, ssboB, W, H, -1);
    if (result == 0)
    {
        printf("computeIFFT2D: execution failed.\n");
        glDeleteBuffers(1, &ssboA);
        glDeleteBuffers(1, &ssboB);
        return 0;
    }

    if (result == ssboA)
        glDeleteBuffers(1, &ssboB);
    else
        glDeleteBuffers(1, &ssboA);

    return result;
}

