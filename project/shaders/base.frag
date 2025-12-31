#version 150

in vec3 pass_Normal;

out vec4 out_Color;

uniform vec3 lightDirection; // in view/eye space

void main(void)
{
	vec3 N = normalize(pass_Normal);
	vec3 L = normalize(lightDirection);
	float lambert = max(dot(N, -L), 0.0);
	vec3 baseColor = vec3(0.8, 0.7, 0.6);
	vec3 ambient = 0.2 * baseColor;
	vec3 diffuse = lambert * baseColor;
	out_Color = vec4(ambient + diffuse, 1.0);
}
