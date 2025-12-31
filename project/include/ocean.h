#pragma once

#include <cstdint>

#include "GL_utilities.h"

constexpr float kDefaultOceanAlpha = 0.04f;
constexpr float kDefaultOceanGamma = 10.3f;

// Minimal 2D vector for spectrum parameterization.
struct OceanVec2
{
	float x;
	float y;
};

// Tunable parameters controlling the initial ocean spectrum.
struct OceanInitParams
{
	float time_scale = 1.0f;   // Speed multiplier for wave evolution
	int resolution = 512;	   // FFT grid resolution (power of two recommended)
	float domainSize = 256.0f; // Physical side length of the simulated patch (meters)
	OceanVec2 windDirection = {1.0f, 0.3f};
	float windSpeed = 30.0f;		  // 10m wind speed in m/s
	float alpha = kDefaultOceanAlpha; // Phillips / PM base spectrum energy scale
	float gamma = kDefaultOceanGamma; // JONSWAP peak amplification
	float spreadExponent = 5.0f;	  // Directional spreading exponent (>0 narrows peak)
	float lowCutoff = 0.0f;			  // Optional soft damping for long waves (rad/m)
	float highCutoff = 1.5f;		  // Optional soft damping for capillaries (rad/m)
	float gravity = 9.81f;			  // Gravitational acceleration (m/s^2)
	float amplitudeScale = 5000.0f;	  // Height amplitude multiplier for water shaders
	float choppiness = 3.0f;		  // Horizontal displacement strength
	uint32_t randomSeed = 132234u;	  // RNG seed for reproducible spectra (0 -> random)
};

// Initialize ocean simulation resources (SSBOs, compute shaders, textures).
void Ocean_Init(OceanInitParams params = OceanInitParams());

// Advance ocean simulation one frame and update height/slope textures.
void Ocean_Update();

// Getters for ocean height and slope textures used by water shader.
GLuint Ocean_GetHeightTexture();
GLuint Ocean_GetSlopeXTexture();
GLuint Ocean_GetSlopeZTexture();
// Getters for horizontal displacement textures (Tessendorf-style choppy waves).
GLuint Ocean_GetDispXTexture();
GLuint Ocean_GetDispZTexture();
GLuint Ocean_GetJacobianTexture();

// Accessors for shader configuration values.
float Ocean_GetPatchSize();
float Ocean_GetAmplitudeScale();
float Ocean_GetChoppiness();

// Access the active initialization parameters.
const OceanInitParams &Ocean_GetParams();
