#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "MicroGlut.h"
#include "GL_utilities.h"
#include "ocean.h"
#include "ocean_spectrum.h"

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440f
#endif

GLuint ssboH0 = 0;
GLuint ssboHt = 0;

int g_fftResolution = 256; // current resolution backing the SSBOs

struct Complex
{
    float r, i;
};

static inline float v2_length(const OceanVec2 &v)
{
    return sqrtf(v.x * v.x + v.y * v.y);
}

static inline OceanVec2 v2_normalize(const OceanVec2 &v)
{
    float l = v2_length(v);
    if (l <= 1e-8f)
        return {0.0f, 0.0f};
    return {v.x / l, v.y / l};
}

static inline float v2_dot(const OceanVec2 &a, const OceanVec2 &b)
{
    return a.x * b.x + a.y * b.y;
}

// JONSWAP spectrum and helpers moved to ocean_spectrum.cpp

void ocean_init(const OceanInitParams &userParams)
{
    OceanInitParams params = userParams;

    g_fftResolution = params.resolution;

    std::vector<Complex> H0(static_cast<size_t>(g_fftResolution) * static_cast<size_t>(g_fftResolution));

    std::mt19937 rng(params.randomSeed ? params.randomSeed : std::random_device{}());
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    const float domain = params.domainSize;
    const float twoPiOverDomain = 2.0f * 3.1415926f / domain;

    OceanVec2 windDirNormalized = v2_normalize(params.windDirection);
    if (v2_length(windDirNormalized) <= 1e-6f)
        windDirNormalized = {1.0f, 0.0f};

    for (int y = 0; y < g_fftResolution; ++y)
    {
        for (int x = 0; x < g_fftResolution; ++x)
        {
            int idx = y * g_fftResolution + x;

            OceanVec2 k = {
                (x - g_fftResolution / 2) * twoPiOverDomain,
                (y - g_fftResolution / 2) * twoPiOverDomain};

            float spectrum = Ocean_JONSWAP_Spectrum(k, params);

            float P = sqrtf(std::max(spectrum, 0.0f));

            float Er = gauss(rng);
            float Ei = gauss(rng);

            H0[idx].r = Er * P * M_SQRT1_2;
            H0[idx].i = Ei * P * M_SQRT1_2;
        }
    }

    // --- Upload buffers ---
    glGenBuffers(1, &ssboH0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboH0);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 sizeof(Complex) * g_fftResolution * g_fftResolution,
                 H0.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &ssboHt);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboHt);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 sizeof(Complex) * g_fftResolution * g_fftResolution,
                 nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}