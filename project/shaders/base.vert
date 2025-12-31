
#version 150

in  vec3 in_Position;
in  vec3 in_Normal;

uniform mat4 projection;
uniform mat4 view;

out vec3 pass_Normal;

void main(void)
{
	// Transform position to clip space
	gl_Position = projection * view * vec4(in_Position, 1.0);

	// Transform normal to view space (assumes no non-uniform scale)
	mat3 normalMatrix = mat3(view);
	pass_Normal = normalize(normalMatrix * in_Normal);
}
