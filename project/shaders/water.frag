#version 150

in vec2 pass_TexCoord;
in vec3 pass_Position;

out vec4 out_Color;

uniform vec3 lightDirection;
uniform vec3 camPos;
uniform float specularStrength = 0.05; // overall specular intensity
uniform float roughness = 0.1; // surface roughness
uniform vec3 baseReflectance = vec3(0.02);
uniform vec3 sunColor = vec3(1.0, 0.82, 0.64); // warmer golden-orange sunlight tint

uniform sampler2D u_SlopeXMap;
uniform sampler2D u_SlopeZMap;
uniform sampler2D u_JacobianMap;
uniform float u_Amplitude;
uniform samplerCube u_Skybox;

uniform vec3 scatteringColor = vec3(0.0, 0.4, 0.5);
uniform vec3 bubbleScatteringColor = vec3(0.0, 0.1, 0.13);
uniform float bubbleDensity = 1.0;
uniform float scatteringStrength = 1.5;
uniform vec3 foamColor;
uniform float foamCompressionStart;
uniform float foamCompressionEnd;
uniform float foamIntensity;

#define PI 3.14159265359

float saturate(float value)
{
	return clamp(value, 0.0, 1.0);
}

vec3 computeSpecular(vec3 normal, vec3 lightDir, vec3 viewDir)
{
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float alpha = roughness * roughness;

    float NdotL = saturate(dot(normal, lightDir));
    float NdotV = saturate(dot(normal, viewDir));
    float NdotH = saturate(dot(normal, halfwayDir));
    float HdotV = saturate(dot(halfwayDir, viewDir));

    // Fresnel
	vec3 F = baseReflectance + (sunColor - baseReflectance) * pow(1.0 - NdotV, 5.0);

    // GGX Normal Distribution
    float D = alpha*alpha / (PI * pow((NdotH*NdotH)*(alpha*alpha-1.0)+1.0, 2.0));

    // Geometry (Smith)
    float k = alpha / 2.0;
    float G = (NdotL / (NdotL*(1.0-k)+k)) * (NdotV / (NdotV*(1.0-k)+k));

	return ((D * G * F) / (4.0 * NdotL * NdotV + 1e-5)) * specularStrength;
}

// Approximates subsurface light scattering in the water volume.
vec3 computeSubsurfaceScattering(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 position)
{
	const float k1 = 1.0; // wavepeak scattering
	const float k2 = 0.5; // scattering based on viewing angle
	const float k3 = 0.8; // direct light scattering
	const float k4 = 0.5; // bubble scattering

	float LdotV = saturate(dot(lightDir, -viewDir));
	float LdotN = saturate(dot(lightDir, normal));
	float VdotN = saturate(dot(viewDir, normal));

	float H = max((position.y + 2.0), 0.0);
	H = pow(H, 2.0);

	vec3 k1Scatter = vec3(1.0) * k1 * H * pow(LdotV, 4.0) * pow(0.5 - 0.5 * LdotN, 3.0);
	vec3 k2Scatter = vec3(1.0) * k2 * pow(VdotN, 2.0);
	vec3 k1k2Scatter = (k1Scatter + k2Scatter) * scatteringColor;

	vec3 k3Scatter = scatteringColor * (k3 * LdotN);
	vec3 k4Scatter = k4 * bubbleScatteringColor * bubbleDensity;

	//return k1Scatter * scatteringColor;

	return k1k2Scatter + k3Scatter + k4Scatter;
}

// Derive per-fragment normal directly from slope textures.
vec3 computeWaveNormal(vec2 uv, out vec2 slopes)
{
	ivec2 texDim = textureSize(u_SlopeXMap, 0);
	if (texDim.x == 0 || texDim.y == 0)
	{
		slopes = vec2(0.0);
		return vec3(0.0, 1.0, 0.0);
	}

	float slopeX = texture(u_SlopeXMap, uv).r;
	float slopeZ = texture(u_SlopeZMap, uv).r;
	slopes = vec2(slopeX, slopeZ);

	vec3 normal = vec3(-u_Amplitude * slopeX, 1.0, -u_Amplitude * slopeZ);
	return normalize(normal);
}

void main(void)
{
	vec3 lightDir = normalize(lightDirection);
	vec2 slopes;
	vec3 normal = computeWaveNormal(pass_TexCoord, slopes);
	vec3 viewDir = normalize(camPos - pass_Position);
	vec3 reflectionDir = reflect(-viewDir, normal);
	vec3 envColor = texture(u_Skybox, reflectionDir).rgb;
	vec3 envColorSun = envColor * (sunColor + vec3(1.0)) / 2;
	envColorSun *= 0.5;
	float NdotV = saturate(dot(normal, viewDir));
	vec3 fresnel = baseReflectance + (vec3(1.0) - baseReflectance) * pow(1.0 - NdotV, 5.0);

	vec3 specular = computeSpecular(normal, lightDir, viewDir);

	vec3 scattering = computeSubsurfaceScattering(normal, viewDir, lightDir, pass_Position) * scatteringStrength;

	vec3 finalColor = specular + scattering;
	finalColor += envColorSun * fresnel;

	float jacobian = texture(u_JacobianMap, pass_TexCoord).r;
	float compression = saturate(1.0 - jacobian);
	float compressionMask = smoothstep(foamCompressionStart, foamCompressionEnd, compression);

	float foamMask = foamIntensity * compressionMask;;
	float foamAmount = saturate(foamMask);
	finalColor = mix(finalColor, foamColor, foamAmount);

	//finalColor = scattering;

	out_Color = vec4(finalColor, 1.0);

}
