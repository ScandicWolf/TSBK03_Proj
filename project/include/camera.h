#pragma once

#include "VectorUtils4.h"

// Initializes camera with default position and orientation.
void Camera_Init();

// Handle input and update camera per frame.
void Camera_HandleInput(float delta);

// Upload current view and related uniforms to shaders.
void Camera_UpdateViewUniforms(GLuint program, GLuint waterProgram, GLuint skyboxProgram, mat4 projection);
