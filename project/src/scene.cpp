#include "scene.h"

#include <stdlib.h>

#include "VectorUtils4.h"

Model *planeModel = nullptr;

Model *CreateSubdividedPlane(int divisions, float size)
{
    int vertsPerSide = divisions + 1;
    int numVertices = vertsPerSide * vertsPerSide;
    int numIndices = divisions * divisions * 6; // two triangles per quad

    vec3 *vertices = (vec3 *)malloc(sizeof(vec3) * numVertices);
    vec3 *normals = (vec3 *)malloc(sizeof(vec3) * numVertices);
    vec2 *texCoords = (vec2 *)malloc(sizeof(vec2) * numVertices);
    GLuint *indices = (GLuint *)malloc(sizeof(GLuint) * numIndices);

    float half = size * 0.5f;
    float step = size / divisions;

    int vi = 0;
    for (int i = 0; i < vertsPerSide; i++)
    {
        for (int j = 0; j < vertsPerSide; j++)
        {
            float x = -half + j * step;
            float z = -half + i * step;
            vertices[vi].x = x;
            vertices[vi].y = 0.0f;
            vertices[vi].z = z;
            normals[vi].x = 0.0f;
            normals[vi].y = 1.0f;
            normals[vi].z = 0.0f;
            float u = (float)j / (float)divisions;
            float v = (float)i / (float)divisions;
            const float repeat = 1.0f;
            texCoords[vi].x = u * repeat;
            texCoords[vi].y = v * repeat;
            vi++;
        }
    }

    int ii = 0;
    for (int i = 0; i < divisions; i++)
    {
        for (int j = 0; j < divisions; j++)
        {
            int v0 = i * vertsPerSide + j;
            int v1 = v0 + 1;
            int v2 = v0 + vertsPerSide;
            int v3 = v2 + 1;
            indices[ii++] = v0;
            indices[ii++] = v2;
            indices[ii++] = v1;
            indices[ii++] = v2;
            indices[ii++] = v3;
            indices[ii++] = v1;
        }
    }

    Model *m = LoadDataToModel(vertices, normals, texCoords, NULL, indices, numVertices, numIndices);
    return m;
}

void Scene_InitModels()
{
    planeModel = CreateSubdividedPlane(500, 128.0f);
}
