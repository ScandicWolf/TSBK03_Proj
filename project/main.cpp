#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <complex>
#include <iostream>
#include "MicroGlut.h"
#define MAIN
#include "VectorUtils4.h"
#include "LittleOBJLoader.h"
#include "GL_utilities.h"
#include "fft_gpu.h"
#include "ocean.h"
#include "camera.h"
#include "scene.h"
#include "LoadTGA.h"

mat4 projection;

// Shader program (make global so callbacks can update uniforms)
GLuint program, waterProgram, skyboxProgram;

// Texture for skybox
GLuint skyboxTexture;

// Vertex Array Object
GLuint skyboxVAO;

float skyboxVertices[] = {
/* skybox cube vertices */
    -1.0f, 1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f,

    -1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f,

    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,

    -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f,

    -1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f,

    -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f};

// Timing for frame delta
double lastTime = 0.0; // seconds

// Forward declarations for callbacks
void Idle(void);
void Reshape(int width, int height);

void skybox_init()
{
    glGenTextures(1, &skyboxTexture);
    glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture);

    TextureData tex_data[6];

    LoadTGATextureData("assets/skybox/skybox_posX.tga", &tex_data[0]);
    LoadTGATextureData("assets/skybox/skybox_negX.tga", &tex_data[1]);
    LoadTGATextureData("assets/skybox/skybox_negY.tga", &tex_data[2]);
    LoadTGATextureData("assets/skybox/skybox_posY.tga", &tex_data[3]);
    LoadTGATextureData("assets/skybox/skybox_posZ.tga", &tex_data[4]);
    LoadTGATextureData("assets/skybox/skybox_negZ.tga", &tex_data[5]);

    for (int i = 0; i < 6; i++)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA,
                     tex_data[i].width, tex_data[i].height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE,
                     tex_data[i].imageData);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glGenVertexArrays(1, &skyboxVAO);
    glBindVertexArray(skyboxVAO);
    GLuint skyboxVBO;
    glGenBuffers(1, &skyboxVBO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glBindVertexArray(0);

    printError("skybox init");
}

void draw_skybox()
{

    glDepthMask(GL_FALSE);
    glUseProgram(skyboxProgram);
    glBindVertexArray(skyboxVAO);
    glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glDepthMask(GL_TRUE);
}

void init(void)
{
    dumpInfo();

    // GL inits
    glClearColor(0.2, 0.2, 0.5, 0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    printError("GL inits");

    // Load and compile shader
    program = loadShaders("shaders/base.vert", "shaders/base.frag");
    waterProgram = loadShaders("shaders/water.vert", "shaders/water.frag");
    skyboxProgram = loadShaders("shaders/skybox.vert", "shaders/skybox.frag");

    printError("init shader");

    // Load models and create plane via scene module
    Scene_InitModels();

    // Set an initial projection matrix (perspective)
    int w = 600, h = 600; // initial window size used in main()
    float aspect = (float)w / (float)h;
    projection = perspective(45.0f, aspect, 0.1f, 1000.0f);
    uploadMat4ToShader(program, "projection", projection);
    uploadMat4ToShader(waterProgram, "projection", projection);

    // Upload initial view matrix
    Camera_UpdateViewUniforms(program, waterProgram, skyboxProgram, projection);

    // Register input callbacks via camera module
    Camera_Init();
    glutIdleFunc(Idle);
    glutReshapeFunc(Reshape);

    // Initialize timing
    lastTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0;

    printError("init arrays");

    // skybox init
    skybox_init();

    // Initialize Tessendorf ocean module (SSBOs, compute shaders, textures)
    Ocean_Init();

    // Bind sampler units and shader uniforms that depend on ocean parameters
    const OceanInitParams &oceanParams = Ocean_GetParams();
    glUseProgram(waterProgram);
    GLint locHM = glGetUniformLocation(waterProgram, "u_HeightMap");
    if (locHM >= 0)
        glUniform1i(locHM, 0);
    GLint locSX = glGetUniformLocation(waterProgram, "u_SlopeXMap");
    if (locSX >= 0)
        glUniform1i(locSX, 1);
    GLint locSZ = glGetUniformLocation(waterProgram, "u_SlopeZMap");
    if (locSZ >= 0)
        glUniform1i(locSZ, 2);
    GLint locDX = glGetUniformLocation(waterProgram, "u_DispX");
    if (locDX >= 0)
        glUniform1i(locDX, 3);
    GLint locDZ = glGetUniformLocation(waterProgram, "u_DispZ");
    if (locDZ >= 0)
        glUniform1i(locDZ, 4);
    GLint locJac = glGetUniformLocation(waterProgram, "u_JacobianMap");
    if (locJac >= 0)
        glUniform1i(locJac, 5);

    GLint locSkybox = glGetUniformLocation(waterProgram, "u_Skybox");
    if (locSkybox >= 0)
        glUniform1i(locSkybox, 6);

    GLint locGrid = glGetUniformLocation(waterProgram, "u_GridSize");
    if (locGrid >= 0)
        glUniform1f(locGrid, oceanParams.domainSize);
    GLint locAmp = glGetUniformLocation(waterProgram, "u_Amplitude");
    if (locAmp >= 0)
        glUniform1f(locAmp, oceanParams.amplitudeScale);

    GLint chopLoc = glGetUniformLocation(waterProgram, "u_Choppiness");
    if (chopLoc >= 0)
        glUniform1f(chopLoc, oceanParams.choppiness);

    const vec3 foamClr = {0.85f, 0.9f, 0.95f};
    GLint locFoamColor = glGetUniformLocation(waterProgram, "foamColor");
    if (locFoamColor >= 0)
        glUniform3f(locFoamColor, foamClr.x, foamClr.y, foamClr.z);

    GLint locFoamCompStart = glGetUniformLocation(waterProgram, "foamCompressionStart");
    if (locFoamCompStart >= 0)
        glUniform1f(locFoamCompStart, 0.15f);

    GLint locFoamCompEnd = glGetUniformLocation(waterProgram, "foamCompressionEnd");
    if (locFoamCompEnd >= 0)
        glUniform1f(locFoamCompEnd, 1.0f);

    GLint locFoamIntensity = glGetUniformLocation(waterProgram, "foamIntensity");
    if (locFoamIntensity >= 0)
        glUniform1f(locFoamIntensity, 1.0f);
    glUseProgram(0);
}

void display(void)
{
    printError("pre display");
    // Advance ocean simulation one frame
    Ocean_Update();

    // clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw skybox first
    draw_skybox();

    // Draw the loaded model if available
    if (planeModel != NULL)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, Ocean_GetHeightTexture());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, Ocean_GetSlopeXTexture());
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, Ocean_GetSlopeZTexture());
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, Ocean_GetDispXTexture());
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, Ocean_GetJacobianTexture());
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, Ocean_GetDispZTexture());
        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTexture);
        glActiveTexture(GL_TEXTURE0);
        DrawModel(planeModel, waterProgram, "in_Position", NULL, "in_TexCoord");
        
    }

    printError("display");

    glutSwapBuffers();
}

void HandleInput(float delta)
{
    Camera_HandleInput(delta);
    Camera_UpdateViewUniforms(program, waterProgram, skyboxProgram, projection);
}

void Idle(void)
{
    // compute delta time (seconds)
    double currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
    float delta = (float)(currentTime - lastTime);
    // clamp large deltas (avoid huge jumps after breakpoint or pause)
    if (delta > 0.1f)
        delta = 0.1f;
    lastTime = currentTime;

    HandleInput(delta);
    // Ensure the scene is redrawn each frame so time-based animations update
    glutPostRedisplay();
}

void Reshape(int width, int height)
{
    if (height == 0)
        height = 1;
    glViewport(0, 0, width, height);
    float aspect = (float)width / (float)height;
    
    mat4 projection = perspective(45.0f, aspect, 0.1f, 1000.0f);
    uploadMat4ToShader(program, "projection", projection);
    uploadMat4ToShader(waterProgram, "projection", projection);
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    // Request an RGBA double-buffered window with a depth buffer
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitContextVersion(4, 3);
    glutInitWindowSize(1200, 1000);
    glutCreateWindow("GL3 white triangle example");
    glutDisplayFunc(display);
    init();
    glutMainLoop();
    return 0;
}