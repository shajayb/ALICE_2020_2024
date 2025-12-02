#define _MAIN_
#ifdef _MAIN_

#include "main.h"
#include "SSAO.h"

SimpleSSAO ssao;
float g_rotation = 0.0f;

SSAOMesh sphereMesh, floorMesh;

// Helper to create random transforms
mat4f getRandomTransform()
{
    float x = (rand() % 800) / 10.0f - 40.0f;
    float y = (rand() % 800) / 10.0f - 40.0f;
    float s = 0.5f + (rand() % 150) / 100.0f;
    float z = s * 1.0f;
    return transform4f(vec3f(x, y, z), vec3f(s, s, s));
}

mat4f getFloorTransform()
{
    return transform4f(vec3f(0, 0, -5), vec3f(120, 120, 0.1f));
}

void createSphereMesh(SSAOMesh& m, float r)
{
    m.vertices.clear(); m.normals.clear(); m.indices.clear();
    for (int i = 0; i <= 32; i++) {
        float v = i / 32.0f, p = v * 3.14159f;
        for (int j = 0; j <= 32; j++) {
            float u = j / 32.0f, t = u * 6.283f;
            float x = sin(p) * cos(t), y = sin(p) * sin(t), z = cos(p);
            m.vertices.insert(m.vertices.end(), { x * r, y * r, z * r });
            m.normals.insert(m.normals.end(), { x, y, z });
        }
    }
    for (int i = 0; i < 32; i++) for (int j = 0; j < 32; j++) {
        int p1 = (i * 33) + j, p2 = p1 + 33;
        m.indices.insert(m.indices.end(), {
            (unsigned)p1, (unsigned)p2, (unsigned)p1 + 1,
            (unsigned)p2, (unsigned)p2 + 1, (unsigned)p1 + 1
            });
    }
    m.dirty = true;
}

void setup()
{
    ssao.setup();
    createSphereMesh(sphereMesh, 1.0f);
    createSphereMesh(floorMesh, 1.0f);

    // --- BUILD SCENE ONCE ---
    ssao.clearQueue();
    ssao.addObject(&floorMesh, getFloorTransform());
    ssao.addObject(&sphereMesh, getRandomTransform());

    resetCamera();
    Alice::setCamera(80, 30, 45, 0, 0);
    glShadeModel(GL_SMOOTH);

    std::cout << "[+] Add Sphere  [R] Reset  [D] Debug" << std::endl;
}

void update(int v)
{
    g_rotation += 0.01f;
}

void draw()
{
    // Sketch controls background color
    backGround(0.95, 0.95, 0.95);

    // Render persistent queue
    ssao.draw();

    // Overlay
    char info[128];
    sprintf(info, "Objs: %d | Mode: %d | Bias: %.2f", ssao.getObjectCount(), ssao.mode, ssao.bias);
    Alice::drawString(info, 10, 20);
    drawGrid(50);
}

void keyPress(unsigned char k, int, int)
{
    if (k == 'r') { setup(); }
    if (k == 'd') ssao.mode = (ssao.mode + 1) % 6;
    if (k == 'b') ssao.blur = !ssao.blur;

    if (k == '+' || k == '=') {
        ssao.addObject(&sphereMesh, getRandomTransform());
    }

    if (k == '1') ssao.bias -= 0.05f; if (k == '2') ssao.bias += 0.05f;
    if (k == '3') ssao.radius -= 0.5f; if (k == '4') ssao.radius += 0.5f;
    if (k == '[') ssao.samples -= 16; if (k == ']') ssao.samples += 16;
}

void mousePress(int, int, int, int) {}
void mouseMotion(int, int) {}

#endif