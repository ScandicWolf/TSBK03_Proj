#include <algorithm>
#include <cmath>

#include "ocean.h"

static inline float v2_length_s(const OceanVec2 &v)
{
    return std::sqrt(v.x * v.x + v.y * v.y);
}

static inline OceanVec2 v2_normalize_s(const OceanVec2 &v)
{
    float l = v2_length_s(v);
    if (l <= 1e-8f)
        return {0.0f, 0.0f};
    return {v.x / l, v.y / l};
}

static inline float v2_dot_s(const OceanVec2 &a, const OceanVec2 &b)
{
    return a.x * b.x + a.y * b.y;
}

static float directionalFactor_s(const OceanVec2 &k, const OceanVec2 &windDirNormalized, float spreadExp)
{
    if (spreadExp <= 0.0f)
        return 1.0f;

    OceanVec2 k_n = v2_normalize_s(k);
    float k_dot_w = v2_dot_s(k_n, windDirNormalized);
    if (k_dot_w <= 0.0f)
        return 0.0f;
    return std::pow(k_dot_w, spreadExp);
}

static float applyCutoffs_s(float energy, float k_len, float lowCutoff, float highCutoff)
{
    if (energy <= 0.0f)
        return 0.0f;

    if (lowCutoff > 0.0f && k_len > 0.0f)
    {
        float ratio = lowCutoff / k_len;
        energy *= std::exp(-ratio * ratio);
    }

    if (highCutoff > 0.0f)
    {
        float ratio = k_len / highCutoff;
        energy *= std::exp(-ratio * ratio);
    }

    return energy;
}

static float computeJONSWAP(const OceanVec2 &k,
                            const OceanVec2 &windDirection,
                            float windSpeed,
                            float alpha,
                            float gamma,
                            float spreadExponent,
                            float lowCutoff,
                            float highCutoff,
                            float gravity)
{
    float k_len = v2_length_s(k);
    if (k_len < 1e-6f || !(windSpeed > 0.0f) || !(std::isfinite(gravity) && gravity > 0.0f))
        return 0.0f;

    const float omega = std::sqrt(gravity * k_len);

    
    const float beta = 1.25f;
    const float g2 = gravity * gravity;
    const float omega_p = 0.877f * gravity / windSpeed;

    const float wp_over_w = omega_p / omega;
    const float pm = alpha * g2 * std::pow(omega, -5.0f) *
                     std::exp(-beta * std::pow(wp_over_w, 4.0f));

    const float sigma = (omega <= omega_p) ? 0.07f : 0.09f;
    const float r = (omega - omega_p) / (sigma * omega_p);
    const float jonswap_peak = std::pow(gamma, std::exp(-0.5f * r * r));

    float S = pm * jonswap_peak;
    S *= 0.5f * std::sqrt(gravity / k_len);
    S = applyCutoffs_s(S, k_len, lowCutoff, highCutoff);

    OceanVec2 windDirNormalized = v2_normalize_s(windDirection);
    if (v2_length_s(windDirNormalized) <= 1e-6f)
        windDirNormalized = {1.0f, 0.0f};

    float dir = directionalFactor_s(k, windDirNormalized, spreadExponent);
    return S * dir;
}

float Ocean_JONSWAP_Spectrum(const OceanVec2 &k,
                             const OceanInitParams &params)
{
    return computeJONSWAP(k,
                          params.windDirection,
                          params.windSpeed,
                          params.alpha,
                          params.gamma,
                          params.spreadExponent,
                          params.lowCutoff,
                          params.highCutoff,
                          params.gravity);
}
