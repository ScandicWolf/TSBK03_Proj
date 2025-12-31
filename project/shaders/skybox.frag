#version 150

in vec3 TexCoords;

uniform samplerCube cubemap;
uniform vec3 lightDir;
uniform vec3 sunColor = vec3(1.0, 0.82, 0.64);

out vec4 out_Color;

void main(void)
{
	vec3 direction = -vec3(TexCoords.x, TexCoords.y, TexCoords.z);
	out_Color = texture(cubemap, direction);

	out_Color.rgb *= (sunColor + vec3(1.0)) / 2; // ambient light

	// fake sun
	float sunIntensity = max(dot(normalize(lightDir), normalize(TexCoords)), 0.0);
	sunIntensity = pow(sunIntensity, 5000.0);
	out_Color.rgb += sunColor * sunIntensity * 5.0;
}
