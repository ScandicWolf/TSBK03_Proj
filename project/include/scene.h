#pragma once

#include "LittleOBJLoader.h"

// Global scene models
extern Model *terrainModel;
extern Model *planeModel;

// Create a subdivided plane centered at origin on the XZ plane (Y=0)
Model *CreateSubdividedPlane(int divisions, float size);

// Load teapot and create plane model
void Scene_InitModels();
