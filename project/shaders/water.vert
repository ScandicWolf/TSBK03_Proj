#version 150

in vec3 in_Position;
in vec2 in_TexCoord;

uniform mat4 projection;
uniform mat4 view;

// FFT output textures
uniform sampler2D u_HeightMap;   // H(x, t) - vertical displacement
uniform sampler2D u_DispX;       // D_x(x, t) - horizontal displacement X
uniform sampler2D u_DispZ;       // D_z(x, t) - horizontal displacement Z
uniform sampler2D u_SlopeXMap;   // ∂h/∂x
uniform sampler2D u_SlopeZMap;   // ∂h/∂z

uniform float u_GridSize;        // world-space size of ocean patch
uniform float u_Amplitude;
uniform float u_Choppiness;      // scales horizontal displacement strength

out vec2 pass_TexCoord;
out vec3 pass_Position;
out vec3 worldPos;

void main()
{
    vec2 uv = in_TexCoord * u_GridSize / 512.0;

    // Height
    float h = texture(u_HeightMap, uv).r * u_Amplitude;

    // Horizontal displacement (must also be scaled)
    float dispX = -texture(u_DispX, uv).r * u_Amplitude * u_Choppiness;
    float dispZ = -texture(u_DispZ, uv).r * u_Amplitude * u_Choppiness;

    // WORLD-SPACE position BEFORE displacement
    vec3 basePos = in_Position;

    // Apply horizontal + vertical displacement
    vec3 displaced = vec3(
        basePos.x + dispX,
        basePos.y + h,
        basePos.z + dispZ
    );

    // Normals (from slopes)
    float sx = texture(u_SlopeXMap, uv).r; 
    float sz = texture(u_SlopeZMap, uv).r;

    pass_Position = displaced;
    worldPos      = displaced;
    pass_TexCoord = uv;

    gl_Position = projection * view * vec4(displaced, 1.0);
}
