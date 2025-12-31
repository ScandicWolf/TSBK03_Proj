#pragma once

// Forward declarations to avoid pulling OpenGL headers for consumers
struct OceanVec2;
struct OceanInitParams;

// Computes the JONSWAP spectrum energy for a single wave-vector.
float Ocean_JONSWAP_Spectrum(const OceanVec2 &k,
                             const OceanInitParams &params);
