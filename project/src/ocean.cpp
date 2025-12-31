#include "ocean.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>

#include "GL_utilities.h"
#include "fft_gpu.h"
#include "LoadTGA.h"

// Forward declarations from ocean_init.cpp (SSBO-based ocean data)
extern void ocean_init(const OceanInitParams &params);
extern GLuint ssboH0; // initial spectrum H0(k)
extern GLuint ssboHt; // spectrum H(k, t)
// No CPU-side height buffer when writing directly to textures

// Active configuration used by the compute pipeline.
static OceanInitParams g_oceanParams{};
static int g_resolution = 0;             // FFT resolution (N)
static float g_patchSize = 0.0f;         // World size of the simulated ocean patch
static float g_gravity = 9.81f;          // Gravity passed to compute shaders
static float g_amplitudeScale = 2500.0f; // Height amplitude scale for water.vert
static float g_choppiness = 2.2f;        // Tessendorf-style horizontal displacement strength

// Compute shader programs
static GLuint evolveProgram = 0;           // evolve spectrum over time
static GLuint extractProgram = 0;          // extract real height / slope from complex field
static GLuint slopeSpecProgram = 0;        // build slope spectra from Ht
static GLuint displacementSpecProgram = 0; // build displacement spectra from Ht
static GLuint jacobianProgram = 0;         // compute jacobian from displacement field

// Textures sampled by water.vert
static GLuint heightTex, slopeXTex, slopeZTex, dispXTex, dispZTex, jacobianTex;
// Expose texture IDs through public API
GLuint Ocean_GetHeightTexture() { return heightTex; }
GLuint Ocean_GetSlopeXTexture() { return slopeXTex; }
GLuint Ocean_GetSlopeZTexture() { return slopeZTex; }
GLuint Ocean_GetDispXTexture() { return dispXTex; }
GLuint Ocean_GetDispZTexture() { return dispZTex; }
GLuint Ocean_GetJacobianTexture() { return jacobianTex; }
float Ocean_GetPatchSize() { return g_patchSize; }
float Ocean_GetAmplitudeScale() { return g_amplitudeScale; }
float Ocean_GetChoppiness() { return g_choppiness; }
const OceanInitParams &Ocean_GetParams() { return g_oceanParams; }

void Texture_Init(GLuint &tex, int width, int height)
{
    std::vector<float> zeros(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, width, height);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, zeros.data());
}

// Converts complex spectra into time-domain floats and writes directly to textures.
static void ExtractToTexture(GLuint timeSSBO, GLuint texture)
{
    if (!extractProgram || !timeSSBO)
        return;

    glUseProgram(extractProgram);
    glUniform1i(glGetUniformLocation(extractProgram, "u_N"), g_resolution);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, timeSSBO);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    const int total = g_resolution * g_resolution;
    const int groups = (total + 256 - 1) / 256;
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
}

// Builds two fields (e.g., Sx/Sz or Dx/Dz) from Ht, IFFT to time domain,
// and writes the real components directly into textures.
static void BuildFieldsFromHt(GLuint computeProgram,
                              GLuint inputHt,
                              GLuint texA, GLuint texB)
{
    if (!computeProgram || !inputHt || g_resolution <= 0)
        return;

    const int total = g_resolution * g_resolution;
    const int groups = (total + 256 - 1) / 256;

    // Allocate temp spectrum SSBOs (complex)
    GLuint ssboSpecA = 0, ssboSpecB = 0;
    glGenBuffers(1, &ssboSpecA);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboSpecA);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Complex) * total, nullptr, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &ssboSpecB);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboSpecB);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Complex) * total, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Build both spectra from Ht
    glUseProgram(computeProgram);
    glUniform1i(glGetUniformLocation(computeProgram, "u_N"), g_resolution);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputHt);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboSpecA);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboSpecB);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // IFFT both spectra to time-domain
    GLuint ssboTimeA = computeIFFT2D(ssboSpecA, g_resolution, g_resolution);
    GLuint ssboTimeB = computeIFFT2D(ssboSpecB, g_resolution, g_resolution);

    // Extract real and upload to textures
    ExtractToTexture(ssboTimeA, texA);
    ExtractToTexture(ssboTimeB, texB);

    // Cleanup
    if (ssboTimeA)
        glDeleteBuffers(1, &ssboTimeA);
    if (ssboTimeB)
        glDeleteBuffers(1, &ssboTimeB);
    glDeleteBuffers(1, &ssboSpecA);
    glDeleteBuffers(1, &ssboSpecB);
}

void SaveTextureToTGA(const char *filename, GLuint TextureID, int width, int height)
{
    if (width <= 0 || height <= 0 || TextureID == 0)
        return;

    // Read back floats from the R32F texture
    std::vector<float> floats(static_cast<size_t>(width) * height);
    glBindTexture(GL_TEXTURE_2D, TextureID);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, floats.data());

    // Find range for normalization (you can clamp to a known range if preferred)
    float vmin = +std::numeric_limits<float>::infinity();
    float vmax = -std::numeric_limits<float>::infinity();
    for (float v : floats)
    {
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
    }
    if (!std::isfinite(vmin) || !std::isfinite(vmax) || vmin == vmax)
    {
        vmin = -1.0f;
        vmax = 1.0f;
    }

    printf("Saving texture to TGA: %s (range %.5f .. %.5f)\n", filename, vmin * g_amplitudeScale, vmax * g_amplitudeScale);
    const float invRange = 1.0f / (vmax - vmin);

    // Convert to 24-bit RGB (replicate grayscale into R=G=B)
    std::vector<GLubyte> rgb(static_cast<size_t>(width) * height * 3);
    for (int i = 0; i < width * height; ++i)
    {
        float norm = (floats[i] - vmin) * invRange; // 0..1
        norm = std::min(std::max(norm, 0.0f), 1.0f);
        GLubyte g8 = static_cast<GLubyte>(norm * 255.0f + 0.5f);
        rgb[3 * i + 0] = g8;
        rgb[3 * i + 1] = g8;
        rgb[3 * i + 2] = g8;
    }

    TextureData texData{};
    texData.width = width;
    texData.height = height;
    texData.bpp = 24; // 24-bit RGB
    texData.imageData = new GLubyte[rgb.size()];
    std::memcpy(texData.imageData, rgb.data(), rgb.size());
    SaveTGA(&texData, const_cast<char *>(filename));
    delete[] texData.imageData;
}

void Ocean_Init(OceanInitParams params)
{
    g_oceanParams = params;
    g_resolution = params.resolution;
    g_patchSize = params.domainSize;
    g_gravity = params.gravity;
    g_amplitudeScale = params.amplitudeScale;
    g_choppiness = params.choppiness;

    if (g_patchSize <= 0.0f)
        g_patchSize = 1.0f;
    if (g_gravity <= 0.0f)
        g_gravity = 9.81f;
    if (g_amplitudeScale <= 0.0f)
        g_amplitudeScale = 1.0f;
    if (g_choppiness < 0.0f)
        g_choppiness = 0.0f;

    g_oceanParams.domainSize = g_patchSize;
    g_oceanParams.gravity = g_gravity;
    g_oceanParams.amplitudeScale = g_amplitudeScale;
    g_oceanParams.choppiness = g_choppiness;

    // Initialize SSBO-based ocean data and compute pipeline (H0/Ht/height buffer)
    ocean_init(g_oceanParams);

    evolveProgram = loadComputeShader("shaders/ocean_evolve.comp");
    extractProgram = loadComputeShader("shaders/ocean_extract_height.comp");
    slopeSpecProgram = loadComputeShader("shaders/ocean_slope_spectrum.comp");
    displacementSpecProgram = loadComputeShader("shaders/ocean_displacement_spectrum.comp");
    jacobianProgram = loadComputeShader("shaders/ocean_jacobian.comp");
    if (!evolveProgram || !extractProgram || !slopeSpecProgram || !displacementSpecProgram || !jacobianProgram)
        std::cout << "Failed to load ocean compute shaders (evolve/extract/slope/displacement/jacobian)\n";

    // Create the height texture used by the vertex shader
    Texture_Init(heightTex, g_resolution, g_resolution);

    // Create slope textures
    Texture_Init(slopeXTex, g_resolution, g_resolution);
    Texture_Init(slopeZTex, g_resolution, g_resolution);

    // Create displacement textures (horizontal choppy displacements)
    Texture_Init(dispXTex, g_resolution, g_resolution);
    Texture_Init(dispZTex, g_resolution, g_resolution);

    // Create jacobian texture
    Texture_Init(jacobianTex, g_resolution, g_resolution);
}

float GetTimeSeconds()
{
    using namespace std::chrono;
    static auto start = steady_clock::now();
    auto now = steady_clock::now();
    return duration<float>(now - start).count();
}

void Ocean_Update()
{
    // 1) Evolve spectrum H(k,t) from H0(k)
    if (evolveProgram && ssboH0 && ssboHt && g_resolution > 0)
    {
        glUseProgram(evolveProgram);
        glUniform1i(glGetUniformLocation(evolveProgram, "u_N"), g_resolution);
        glUniform1f(glGetUniformLocation(evolveProgram, "u_domainSize"), g_patchSize);
        glUniform1f(glGetUniformLocation(evolveProgram, "u_gravity"), g_gravity);

        float t = GetTimeSeconds() * g_oceanParams.time_scale;
        glUniform1f(glGetUniformLocation(evolveProgram, "u_time"), t);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboH0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboHt);
        int total = g_resolution * g_resolution;
        int groups = (total + 256 - 1) / 256; // local_size_x = 256
        glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // 2) Inverse 2D FFT to time-domain heights (complex)
    GLuint timeSSBO = 0;
    if (ssboHt && g_resolution > 0)
    {
        timeSSBO = computeIFFT2D(ssboHt, g_resolution, g_resolution);
    }

    // 3) Extract real height and upload to texture directly on the GPU
    if (timeSSBO && extractProgram)
    {
        ExtractToTexture(timeSSBO, heightTex);
        // SaveTextureToTGA("./out/ocean_height.tga", heightTex, g_resolution, g_resolution);
    }
    if (timeSSBO)
    {
        glDeleteBuffers(1, &timeSSBO);
    }

    // 4) Build slope fields Sx/Sz and upload
    if (ssboHt && g_resolution > 0)
    {
        BuildFieldsFromHt(slopeSpecProgram, ssboHt, slopeXTex, slopeZTex);
    }

    // 5) Build horizontal displacement fields Dx/Dz and upload
    if (ssboHt && g_resolution > 0)
    {
        BuildFieldsFromHt(displacementSpecProgram, ssboHt, dispXTex, dispZTex);
    }

    // 6) Compute jacobian determinant texture from displaced field
    if (jacobianProgram && g_resolution > 0)
    {
        glUseProgram(jacobianProgram);
        float cellSize = g_patchSize / static_cast<float>(g_resolution);
        glUniform1f(glGetUniformLocation(jacobianProgram, "u_CellSize"), cellSize);
        glUniform1f(glGetUniformLocation(jacobianProgram, "u_Amplitude"), g_amplitudeScale);
        glUniform1f(glGetUniformLocation(jacobianProgram, "u_Choppiness"), g_choppiness);
        glUniform1i(glGetUniformLocation(jacobianProgram, "u_DispX"), 0);
        glUniform1i(glGetUniformLocation(jacobianProgram, "u_DispZ"), 1);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, dispXTex);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, dispZTex);

        glBindImageTexture(0, jacobianTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

        int groups = (g_resolution + 15) / 16;
        glDispatchCompute(groups, groups, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

        // SaveTextureToTGA("./out/ocean_jacobian.tga", jacobianTex, g_resolution, g_resolution);

        glActiveTexture(GL_TEXTURE0);
    }
}
