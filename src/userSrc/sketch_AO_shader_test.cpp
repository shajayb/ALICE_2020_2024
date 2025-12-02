#define _MAIN_
#ifdef _MAIN_

#include "main.h"
#include "SSAO.h"

SimpleSSAO ssao;
float g_rotation = 0.0f;

// Container for dynamic objects
struct SceneObject {
    vec3f pos;
    float scale;
    SSAOMesh* meshPtr;
};

std::vector<SceneObject> sceneObjects;
SSAOMesh sphereMesh, floorMesh;

// Helper to build non-uniform scaling matrix for floor
mat4f getFloorTransform() {
    mat4f M = identity4f();
    M.m[0] = 120.0f; M.m[5] = 120.0f; M.m[10] = 0.1f; // Scale
    M.m[14] = -5.0f; // Translate Z
    return M;
}

void createSphereMesh(SSAOMesh& m, float r) {
    m.vertices.clear(); m.normals.clear(); m.indices.clear();
    for (int i = 0; i <= 32; i++) {
        float v = i / 32.0f, p = v * 3.14159f; for (int j = 0; j <= 32; j++) {
            float u = j / 32.0f, t = u * 6.283f;
            float x = sin(p) * cos(t), y = sin(p) * sin(t), z = cos(p);
            m.vertices.insert(m.vertices.end(), { x * r,y * r,z * r });
            m.normals.insert(m.normals.end(), { x,y,z });
        }
    }
    for (int i = 0; i < 32; i++) for (int j = 0; j < 32; j++) {
        int p1 = (i * 33) + j, p2 = p1 + 33;
        m.indices.insert(m.indices.end(), { (unsigned)p1,(unsigned)p2,(unsigned)p1 + 1, (unsigned)p2,(unsigned)p2 + 1,(unsigned)p1 + 1 });
    }
    m.dirty = true;
}

void addRandomSphere() {
    SceneObject obj;
    obj.meshPtr = &sphereMesh;
    float x = (rand() % 800) / 10.0f - 40.0f;
    float y = (rand() % 800) / 10.0f - 40.0f;
    obj.scale = 0.5f + (rand() % 150) / 100.0f;
    obj.pos = { x, y, obj.scale * 1.0f }; // Sit on floor
    sceneObjects.push_back(obj);
}

void setup() {
    ssao.setup();
    createSphereMesh(sphereMesh, 1.0f);
    createSphereMesh(floorMesh, 1.0f);

    addRandomSphere();
    resetCamera(); Alice::setCamera(80, 30, 45, 0, 0);
    glShadeModel(GL_SMOOTH);
}

void update(int v) { g_rotation += 0.01f; }

void draw() {
    // 1. Reset Queue
    ssao.clearQueue();

    // 2. Add Floor (Matrix transform)
    ssao.addObject(&floorMesh, getFloorTransform());

    // 3. Add Dynamic Spheres (Matrix transform from pos/scale)
    for (auto& obj : sceneObjects) {
        mat4f M = transform4f(obj.pos, vec3f{ obj.scale, obj.scale, obj.scale });
        ssao.addObject(obj.meshPtr, M);
    }

    // 4. Render
    ssao.draw();

    // 5. UI
    char info[128];
    sprintf(info, "Objs:%d | Mode:%d | Bias:%.2f", (int)sceneObjects.size(), ssao.mode, ssao.bias);
    Alice::drawString(info, 10, 20);
    drawGrid(50);
}

void keyPress(unsigned char k, int, int) {
    if (k == 'r') { sceneObjects.clear(); setup(); }
    if (k == 'd') ssao.mode = (ssao.mode + 1) % 6;
    if (k == 'b') ssao.blur = !ssao.blur;
    if (k == '+' || k == '=') addRandomSphere();
    if (k == '1') ssao.bias -= 0.05f; if (k == '2') ssao.bias += 0.05f;
    if (k == '3') ssao.radius -= 0.5f; if (k == '4') ssao.radius += 0.5f;
    if (k == '[') ssao.samples -= 16; if (k == ']') ssao.samples += 16;
}
void mousePress(int, int, int, int) {} void mouseMotion(int, int) {}

#endif