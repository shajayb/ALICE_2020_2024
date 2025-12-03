#define _MAIN_
#ifdef _MAIN_

#include "main.h"
#include "SSAO.h"

SimpleSSAO ssao;

SSAOMesh sphereMesh, floorMesh;
SSAOMesh orientedCubeMesh;

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

void createOrientedCube(SSAOMesh& m, vec3f dir, float L, float W, float D)
{
    m.vertices.clear(); m.normals.clear(); m.indices.clear();
    float x = L * 0.5f, y = W * 0.5f, z = D * 0.5f;

    std::vector<vec3f> v = {
        {-x,-y,-z},{ x,-y,-z},{ x, y,-z},{-x, y,-z}, // Back
        {-x,-y, z},{ x,-y, z},{ x, y, z},{-x, y, z}, // Front
        {-x,-y,-z},{-x,-y, z},{-x, y, z},{-x, y,-z}, // Left
        { x,-y,-z},{ x,-y, z},{ x, y, z},{ x, y,-z}, // Right
        {-x, y,-z},{ x, y,-z},{ x, y, z},{-x, y, z}, // Top
        {-x,-y,-z},{ x,-y,-z},{ x,-y, z},{-x,-y, z}  // Bottom
    };

    std::vector<vec3f> n = {
        { 0, 0,-1},{ 0, 0,-1},{ 0, 0,-1},{ 0, 0,-1},
        { 0, 0, 1},{ 0, 0, 1},{ 0, 0, 1},{ 0, 0, 1},
        {-1, 0, 0},{-1, 0, 0},{-1, 0, 0},{-1, 0, 0},
        { 1, 0, 0},{ 1, 0, 0},{ 1, 0, 0},{ 1, 0, 0},
        { 0, 1, 0},{ 0, 1, 0},{ 0, 1, 0},{ 0, 1, 0},
        { 0,-1, 0},{ 0,-1, 0},{ 0,-1, 0},{ 0,-1, 0}
    };

    mat4f R = alignToDir(dir);

    for (int i = 0; i < 24; i++) {
        // Vertex Rotation
        float vx = R.m[0] * v[i].x + R.m[4] * v[i].y + R.m[8] * v[i].z;
        float vy = R.m[1] * v[i].x + R.m[5] * v[i].y + R.m[9] * v[i].z;
        float vz = R.m[2] * v[i].x + R.m[6] * v[i].y + R.m[10] * v[i].z;
        m.vertices.insert(m.vertices.end(), { vx, vy, vz });

        // Normal Rotation
        float nx = R.m[0] * n[i].x + R.m[4] * n[i].y + R.m[8] * n[i].z;
        float ny = R.m[1] * n[i].x + R.m[5] * n[i].y + R.m[9] * n[i].z;
        float nz = R.m[2] * n[i].x + R.m[6] * n[i].y + R.m[10] * n[i].z;
        m.normals.insert(m.normals.end(), { nx, ny, nz });
    }

    unsigned int idx[] = { 0,2,1, 0,3,2, 4,5,6, 4,6,7, 8,9,10, 8,10,11, 12,14,13, 12,15,14, 16,17,18, 16,18,19, 20,22,21, 20,23,22 };
    for (int i = 0; i < 36; i++) m.indices.push_back(idx[i]);
    m.dirty = true;
}

void addVectorField(SSAOMesh* mesh, int count) {
    for (int i = 0; i < count; i++) {
        float t = i * 0.2f;
        float r = 10.0f + i * 1.5f;

        // Spiral Pattern
        vec3f pos(cos(t) * r, sin(t) * r, 5.0f + i * 0.5f);

        // Tangent Direction
        vec3f dir(-sin(t), cos(t), 0.2f);

        // Align Mesh to Dir using Matrix
        mat4f M = alignToDir(dir);
        M.m[12] = pos.x; M.m[13] = pos.y; M.m[14] = pos.z;

        ssao.addObject(mesh, M);
    }
}

void addRandomSphere() {
    // Use existing sphereMesh, just transform it
    float x = (rand() % 800) / 10.0f - 40.0f;
    float y = (rand() % 800) / 10.0f - 40.0f;
    float z = ofRandom(1, 30);
    float s = ofRandom(1, 5);

    mat4f M = transform4f(vec3f(x, y, z), vec3f(s, s, s));
    ssao.addObject(&sphereMesh, M);
}

void setup()
{
    ssao.setup();
    createSphereMesh(sphereMesh, 1.0f);

    // --- FIX: Use a Cube for the floor instead of a Sphere ---
    // A flattened sphere has unstable normals at the edges, causing SSAO artifacts.
    // A box scaled to be a floor has perfectly flat normals (0,0,1).
    createOrientedCube(floorMesh, vec3f(0, 0, 1), 1.0f, 1.0f, 1.0f);

    // Create a Long thin slab for vector field
    createOrientedCube(orientedCubeMesh, vec3f(1, 0, 0), 6.0f, 2.0f, 1.0f);

    ssao.clearQueue();

    // 1. Floor (Scale the Unit Cube to be a floor)
    mat4f floorM = transform4f(vec3f(0, 0, -3), vec3f(120, 120,2.f));
    ssao.addObject(&floorMesh, floorM);

    // 2. Vector Field of Cubes
    addVectorField(&orientedCubeMesh, 40);

    // 3. Initial Sphere
    addRandomSphere();

    glShadeModel(GL_SMOOTH);
}

void update(int v) {  }

void draw()
{
    backGround(0.95, 0.95, 0.95);

    // Queue is persistent, just draw.
    ssao.draw();

    drawGrid(50);

    //--------------------------------
    setup2d();
    char info[128];
    const char* mNames[] = { "LIT", "AO_RAW", "AO_BLUR", "NORM", "DEPTH", "POS", "DELTA", "SAMPLES" };
    sprintf(info, "Objs:%d | Mode:%s | Bias:%.2f | Rad:%.2f |Samps:%d", ssao.getObjectCount(), mNames[ssao.mode % 8], ssao.bias, ssao.radius, ssao.samples);
    glColor3f(0, 0, 0);
    Alice::drawString(info, 20, 40);
    restore3d();
}

void keyPress(unsigned char k, int, int) {
    if (k == 'r') { ssao.clearQueue(); setup(); }
    if (k == 'd') ssao.mode = (ssao.mode + 1) % 8;
    if (k == 'b') ssao.blur = !ssao.blur;
    if (k == '+' || k == '=') addRandomSphere();
    if (k == '1') ssao.bias -= 0.05f; if (k == '2') ssao.bias += 0.05f;
    if (k == '3') ssao.radius -= 0.5f; if (k == '4') ssao.radius += 0.5f;
    if (k == '[') ssao.samples -= 16; if (k == ']') ssao.samples += 16;
}
void mousePress(int, int, int, int) {} void mouseMotion(int, int) {}

#endif