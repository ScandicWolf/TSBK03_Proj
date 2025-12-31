#include "camera.h"

#include "GL_utilities.h"
#include "MicroGlut.h"

// Internal camera state
static vec3 camPos = {0.0f, 2.0f, 6.0f};
static vec3 camFront = {0.0f, 0.0f, -1.0f};
static vec3 camUp = {0.0f, 1.0f, 0.0f};
static float yaw = -90.0f;
static float pitch = 0.0f;
static float rotationSpeed = 120.0f;
static float moveSpeed = 15.0f;

static bool keyStates[256] = {false};

static void KeyDown(unsigned char key, int x, int y)
{
    if (key == 0x1b)
        exit(0);
    keyStates[(unsigned char)key] = true;
}

static void KeyUp(unsigned char key, int x, int y)
{
    keyStates[(unsigned char)key] = false;
}

void Camera_Init()
{
    glutKeyboardFunc(KeyDown);
    glutKeyboardUpFunc(KeyUp);
}

void Camera_HandleInput(float delta)
{
    bool moved = false;
    vec3 right = normalize(cross(camFront, camUp));
    vec3 moveDir = SetVec3(0.0f, 0.0f, 0.0f);

    if (keyStates[(unsigned char)'w'])
    {
        moveDir += camFront;
        moved = true;
    }
    if (keyStates[(unsigned char)'s'])
    {
        moveDir -= camFront;
        moved = true;
    }
    if (keyStates[(unsigned char)'a'])
    {
        moveDir -= right;
        moved = true;
    }
    if (keyStates[(unsigned char)'d'])
    {
        moveDir += right;
        moved = true;
    }
    if (keyStates[(unsigned char)'q'])
    {
        moveDir += camUp;
        moved = true;
    }
    if (keyStates[(unsigned char)'e'])
    {
        moveDir -= camUp;
        moved = true;
    }

    camPos = camPos + moveDir * moveSpeed * delta;

    if (keyStates[(unsigned char)'j'])
    {
        yaw -= rotationSpeed * delta;
        moved = true;
    }
    if (keyStates[(unsigned char)'l'])
    {
        yaw += rotationSpeed * delta;
        moved = true;
    }
    if (keyStates[(unsigned char)'i'])
    {
        pitch += rotationSpeed * delta;
        moved = true;
    }
    if (keyStates[(unsigned char)'k'])
    {
        pitch -= rotationSpeed * delta;
        moved = true;
    }

    if (moved)
    {
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
        float radYaw = yaw * (M_PI / 180.0f);
        float radPitch = pitch * (M_PI / 180.0f);
        vec3 front = SetVec3(cos(radYaw) * cos(radPitch),
                             sin(radPitch),
                             sin(radYaw) * cos(radPitch));
        camFront = normalize(front);
    }
}

void Camera_UpdateViewUniforms(GLuint program, GLuint waterProgram, GLuint skyboxProgram, mat4 projection)
{
    mat4 view = lookAt(camPos, camPos + camFront, camUp);

    mat4 noTranslationView = mat4(mat3(view)); // copy rotation part only

    uploadMat4ToShader(program, "view", view);
    uploadMat4ToShader(waterProgram, "view", view);
    uploadMat4ToShader(skyboxProgram, "view", noTranslationView);
    uploadMat4ToShader(skyboxProgram, "projection", projection);

    vec3 lightWorld = SetVec3(3.0f, 1.0f, 1.0f);
    lightWorld = normalize(lightWorld);
    vec3 lightView = MultVec3(view, lightWorld);
    uploadUniformVec3ToShader(program, "lightDirection", lightView);
    uploadUniformVec3ToShader(waterProgram, "lightDirection", lightWorld);
    uploadUniformVec3ToShader(waterProgram, "camPos", camPos);
    uploadUniformVec3ToShader(skyboxProgram, "lightDir", lightWorld);
}
