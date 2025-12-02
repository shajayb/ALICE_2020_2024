#define _MAIN_
#ifdef _MAIN_

#include "main.h"
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <cmath>

using namespace std;

// --------------------------------------------------------------------------------
// Globals
// --------------------------------------------------------------------------------
float g_bias = 0.2f;
float g_radius = 5.0f;
int   g_debugMode = 1; // Start in AO_RAW
float g_rotation = 0.0f;
bool  g_enableBlur = true;

// --------------------------------------------------------------------------------
// Math Helpers
// --------------------------------------------------------------------------------
const float PI_F = 3.14159265358979f;
struct Mat4 { float m[16]; };

Mat4 identity() { return { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }; }

Mat4 multiply(Mat4 A, Mat4 B) {
    Mat4 R;
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            R.m[c * 4 + r] = A.m[0 * 4 + r] * B.m[c * 4 + 0] + A.m[1 * 4 + r] * B.m[c * 4 + 1] + A.m[2 * 4 + r] * B.m[c * 4 + 2] + A.m[3 * 4 + r] * B.m[c * 4 + 3];
    return R;
}

Mat4 perspective(float fov, float aspect, float znear, float zfar) {
    float f = 1.0f / tan(fov * 0.5f * PI_F / 180.0f);
    Mat4 R = { 0 };
    R.m[0] = f / aspect; R.m[5] = f;
    R.m[10] = (zfar + znear) / (znear - zfar);
    R.m[11] = -1.0f;
    R.m[14] = (2.0f * zfar * znear) / (znear - zfar);
    return R;
}

float lerp(float a, float b, float f) { return a + f * (b - a); }

// --------------------------------------------------------------------------------
// FBO
// --------------------------------------------------------------------------------
struct FBO {
    GLuint id = 0; int w = 0, h = 0; vector<GLuint> textures;
    void resize(int _w, int _h, int n) {
        if (w == _w && h == _h) return;
        if (id) { glDeleteFramebuffers(1, &id); glDeleteTextures(textures.size(), textures.data()); textures.clear(); }
        w = _w; h = _h;
        glGenFramebuffers(1, &id); glBindFramebuffer(GL_FRAMEBUFFER, id);
        vector<GLenum> dbs;
        for (int i = 0; i < n; i++) {
            GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, t, 0);
            textures.push_back(t); dbs.push_back(GL_COLOR_ATTACHMENT0 + i);
        }
        GLuint r; glGenRenderbuffers(1, &r); glBindRenderbuffer(GL_RENDERBUFFER, r);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, r);
        glDrawBuffers(dbs.size(), dbs.data());

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) cout << "FBO Error" << endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    void bind() { glBindFramebuffer(GL_FRAMEBUFFER, id); glViewport(0, 0, w, h); glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); }
    void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }
};
FBO gBuffer; FBO ssaoFBO;

// --------------------------------------------------------------------------------
// SHADERS
// --------------------------------------------------------------------------------
class GeomShader {
public:
    GLuint pid;
    void init() {
        string vs = R"(
            #version 400
            layout(location=0) in vec3 P; layout(location=1) in vec3 N;
            uniform mat4 MVP, MV; out vec3 vP, vN;
            void main(){ 
                vP = (MV * vec4(P,1)).xyz; 
                vN = normalize(transpose(inverse(mat3(MV))) * N);
                gl_Position = MVP * vec4(P,1); 
            }
        )";
        string fs = R"(
            #version 400
            layout(location=0) out vec4 gP; layout(location=1) out vec4 gN;
            in vec3 vP, vN;
            void main(){ gP=vec4(vP,1); gN=vec4(normalize(vN),1); }
        )";
        create(vs, fs);
    }
    void bind(Mat4& MVP, Mat4& MV) { glUseProgram(pid); glUniformMatrix4fv(glGetUniformLocation(pid, "MVP"), 1, 0, MVP.m); glUniformMatrix4fv(glGetUniformLocation(pid, "MV"), 1, 0, MV.m); }
    void create(string v, string f) {
        GLuint p = glCreateProgram(), vs = glCreateShader(GL_VERTEX_SHADER), fs = glCreateShader(GL_FRAGMENT_SHADER);
        const char* s = v.c_str(); glShaderSource(vs, 1, &s, 0); glCompileShader(vs);
        s = f.c_str(); glShaderSource(fs, 1, &s, 0); glCompileShader(fs);
        glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p); pid = p;
    }
};

class SSAOShader {
public:
    GLuint pid; float kernel[192];
    void init() {
        uniform_real_distribution<float> rnd(0, 1); default_random_engine gen;
        for (int i = 0; i < 64; ++i) {
            float s = (float)i / 64.0; s = 0.1 + 0.9 * s * s;
            kernel[i * 3 + 0] = (rnd(gen) * 2 - 1) * s; kernel[i * 3 + 1] = (rnd(gen) * 2 - 1) * s; kernel[i * 3 + 2] = (rnd(gen)) * s;
        }
        string fs = R"(
            #version 400
            out float Occ; in vec2 UV;
            uniform sampler2D gP, gN; uniform vec3 k[64]; uniform mat4 P;
            uniform float rad, bias, width, height;
            float rand(vec2 n){return fract(sin(dot(n,vec2(12.9898,4.1414)))*43758.5453);}
            void main(){
                vec4 pDat = texture(gP, UV);
                if(pDat.a < 0.5) { Occ=1.0; return; } // Background
                vec3 pos = pDat.xyz;
                vec3 norm = normalize(texture(gN, UV).rgb);
                
                vec2 nUV = UV * vec2(width, height) / 4.0;
                vec3 rv = normalize(vec3(rand(nUV), rand(nUV+0.1), 0));
                vec3 T = normalize(rv - norm * dot(rv, norm));
                vec3 B = cross(norm, T);
                mat3 TBN = mat3(T, B, norm);

                float occ = 0.0;
                for(int i=0; i<64; ++i) {
                    vec3 sPos = pos + (TBN * k[i]) * rad;
                    vec4 off = P * vec4(sPos, 1.0);
                    off.xyz /= off.w; off.xyz = off.xyz * 0.5 + 0.5;
                    
                    if(off.x<0||off.x>1||off.y<0||off.y>1) continue;
                    
                    float d = texture(gP, off.xy).z;
                    float range = smoothstep(0.0, 1.0, rad / abs(pos.z - d));
                    if (d >= sPos.z + bias) occ += 1.0 * range;
                }
                Occ = 1.0 - (occ / 64.0);
            }
        )";
        string vs = "#version 400\n layout(location=0) in vec3 P; layout(location=1) in vec2 T; out vec2 UV; void main(){UV=T; gl_Position=vec4(P,1);}";
        create(vs, fs);
    }
    void bind(GLuint p, GLuint n, Mat4& proj, int w, int h) {
        glUseProgram(pid);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, p); glUniform1i(glGetUniformLocation(pid, "gP"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, n); glUniform1i(glGetUniformLocation(pid, "gN"), 1);
        glUniform3fv(glGetUniformLocation(pid, "k"), 64, kernel);
        glUniformMatrix4fv(glGetUniformLocation(pid, "P"), 1, 0, proj.m);
        glUniform1f(glGetUniformLocation(pid, "rad"), g_radius);
        glUniform1f(glGetUniformLocation(pid, "bias"), g_bias);
        glUniform1f(glGetUniformLocation(pid, "width"), (float)w);
        glUniform1f(glGetUniformLocation(pid, "height"), (float)h);
    }
    void create(string v, string f) {
        GLuint p = glCreateProgram(), vs = glCreateShader(GL_VERTEX_SHADER), fs = glCreateShader(GL_FRAGMENT_SHADER);
        const char* s = v.c_str(); glShaderSource(vs, 1, &s, 0); glCompileShader(vs); s = f.c_str(); glShaderSource(fs, 1, &s, 0); glCompileShader(fs);
        glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p); pid = p;
    }
};

class BlurShader {
public:
    GLuint pid;
    void init() {
        string fs = R"(
            #version 400
            out vec4 C; in vec2 UV;
            uniform sampler2D sIn, gP, gN; uniform int mode; uniform bool blur; uniform float width, height;
            void main(){
                float occ = texture(sIn, UV).r;
                if(blur) {
                    vec2 ts = 1.0/vec2(width,height);
                    float res=0;
                    for(int x=-2;x<2;x++) for(int y=-2;y<2;y++) res+=texture(sIn, UV+vec2(x,y)*ts).r;
                    occ = res/16.0;
                }
                occ = pow(occ, 3.0); 
                vec4 pD = texture(gP, UV);
                if(pD.a < 0.5) { C=vec4(0.9); return; }

                if(mode==1) { C=vec4(vec3(texture(sIn,UV).r),1); return; } // RAW
                if(mode==2) { C=vec4(vec3(occ),1); return; } // BLUR
                if(mode==3) { C=vec4(texture(gN,UV).rgb*0.5+0.5,1); return; } // NORM
                if(mode==4) { C=vec4(vec3(clamp((-pD.z-20)/100,0,1)),1); return; } // DEPTH
                if(mode==5) { C=vec4(abs(pD.xyz)/50,1); return; } // POS
                if(mode==6) { float d = 1.0-texture(sIn,UV).r; C=vec4(d,0,0,1); return; } // DELTA (Red=Occluded)

                vec3 L=normalize(vec3(0.5,0.5,1));
                float diff=max(dot(texture(gN,UV).rgb,L),0);
                C=vec4(vec3(0.3*occ + 0.7*diff),1);
            }
        )";
        string vs = "#version 400\n layout(location=0) in vec3 P; layout(location=1) in vec2 T; out vec2 UV; void main(){UV=T; gl_Position=vec4(P,1);}";
        create(vs, fs);
    }
    void bind(GLuint s, GLuint p, GLuint n, int w, int h) {
        glUseProgram(pid);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, s); glUniform1i(glGetUniformLocation(pid, "sIn"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, p); glUniform1i(glGetUniformLocation(pid, "gP"), 1);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, n); glUniform1i(glGetUniformLocation(pid, "gN"), 2);
        glUniform1i(glGetUniformLocation(pid, "mode"), g_debugMode);
        glUniform1i(glGetUniformLocation(pid, "blur"), g_enableBlur);
        glUniform1f(glGetUniformLocation(pid, "width"), (float)w);
        glUniform1f(glGetUniformLocation(pid, "height"), (float)h);
    }
    void create(string v, string f) {
        GLuint p = glCreateProgram(), vs = glCreateShader(GL_VERTEX_SHADER), fs = glCreateShader(GL_FRAGMENT_SHADER);
        const char* s = v.c_str(); glShaderSource(vs, 1, &s, 0); glCompileShader(vs); s = f.c_str(); glShaderSource(fs, 1, &s, 0); glCompileShader(fs);
        glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p); pid = p;
    }
};

// --------------------------------------------------------------------------------
// MESHES
// --------------------------------------------------------------------------------
struct Quad {
    GLuint vao;
    void init() {
        float q[] = { -1,1,0,0,1, -1,-1,0,0,0, 1,1,0,1,1, 1,-1,0,1,0 };
        glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        GLuint v; glGenBuffers(1, &v); glBindBuffer(GL_ARRAY_BUFFER, v); glBufferData(GL_ARRAY_BUFFER, sizeof(q), q, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, 0, 20, 0); glEnableVertexAttribArray(0); glVertexAttribPointer(1, 2, GL_FLOAT, 0, 20, (void*)12); glEnableVertexAttribArray(1);
    }
    void draw() { glBindVertexArray(vao); glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); }
};
struct Sphere {
    GLuint vao; int cnt;
    void init(float r, int sl, int st) {
        vector<float> d; vector<unsigned> idx;
        for (int i = 0; i <= st; i++) {
            float v = i / (float)st, phi = v * PI_F;
            for (int j = 0; j <= sl; j++) {
                float u = j / (float)sl, th = u * PI_F * 2;
                float x = sin(phi) * cos(th), y = sin(phi) * sin(th), z = cos(phi);
                d.insert(d.end(), { x * r,y * r,z * r, x,y,z });
            }
        }
        for (int i = 0; i < st; i++) for (int j = 0; j < sl; j++) {
            int p1 = (i * (sl + 1)) + j, p2 = p1 + sl + 1;
            idx.insert(idx.end(), { (unsigned)p1,(unsigned)p2,(unsigned)p1 + 1, (unsigned)p2,(unsigned)p2 + 1,(unsigned)p1 + 1 });
        }
        cnt = idx.size();
        glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        GLuint vb, eb; glGenBuffers(1, &vb); glBindBuffer(GL_ARRAY_BUFFER, vb); glBufferData(GL_ARRAY_BUFFER, d.size() * 4, d.data(), GL_STATIC_DRAW);
        glGenBuffers(1, &eb); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eb); glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * 4, idx.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, 0, 24, 0); glEnableVertexAttribArray(0); glVertexAttribPointer(1, 3, GL_FLOAT, 0, 24, (void*)12); glEnableVertexAttribArray(1);
    }
    void draw() { glBindVertexArray(vao); glDrawElements(GL_TRIANGLES, cnt, GL_UNSIGNED_INT, 0); }
};

GeomShader gs; SSAOShader ss; BlurShader bs; Quad quad; Sphere sphere; Sphere floorObj;

void setup() {
    gs.init(); ss.init(); bs.init(); quad.init();
    sphere.init(5.0f, 32, 32); floorObj.init(80.0f, 64, 64);
    resetCamera(); Alice::setCamera(120, 30, 45, 0, 0);
    glShadeModel(GL_SMOOTH);
}

void update(int v) { g_rotation += 0.01f; }

void draw() {
    int w = glutGet(GLUT_WINDOW_WIDTH); int h = glutGet(GLUT_WINDOW_HEIGHT);
    gBuffer.resize(w, h, 2); ssaoFBO.resize(w, h, 1);

    Mat4 P = perspective(60.0f, (float)w / h, 1.0f, 500.0f);
    float mr[16];
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();
    Alice::updateCamera(); glGetFloatv(GL_MODELVIEW_MATRIX, mr);
    Mat4 V; memcpy(V.m, mr, 16 * 4);

    // --- 1. G-BUFFER ---
    // CRITICAL FIX: DISABLE BLEND TO WRITE ALPHA=1.0 CORRECTLY
    glDisable(GL_BLEND);
    gBuffer.bind(); glEnable(GL_DEPTH_TEST);

    // Floor
    glPushMatrix(); glTranslatef(0, 0, -10); glScalef(1, 1, 0.05);
    float mf[16]; glGetFloatv(GL_MODELVIEW_MATRIX, mf);
    Mat4 Mf; memcpy(Mf.m, mf, 16 * 4); Mat4 MVPf = multiply(P, Mf);
    gs.bind(MVPf, Mf); floorObj.draw(); glPopMatrix();

    // Spheres
    for (int x = -2; x <= 2; x++) for (int y = -2; y <= 2; y++) {
        glPushMatrix(); glTranslatef(x * 15.0f, y * 15.0f, 0);
        float ms[16]; glGetFloatv(GL_MODELVIEW_MATRIX, ms);
        Mat4 Ms; memcpy(Ms.m, ms, 16 * 4); Mat4 MVPs = multiply(P, Ms);
        gs.bind(MVPs, Ms); sphere.draw(); glPopMatrix();
    }
    gBuffer.unbind();

    // --- 2. SSAO ---
    ssaoFBO.bind(); glDisable(GL_DEPTH_TEST);
    ss.bind(gBuffer.textures[0], gBuffer.textures[1], P, w, h);
    quad.draw();
    ssaoFBO.unbind();

    // --- 3. BLUR/LIT ---
    // Re-enable blend for UI overlay, but here we overwrite screen
    glDisable(GL_BLEND);
    glClearColor(1, 1, 1, 1); glClear(GL_COLOR_BUFFER_BIT);
    bs.bind(ssaoFBO.textures[0], gBuffer.textures[0], gBuffer.textures[1], w, h);
    quad.draw();

    // Cleanup
    glPopMatrix(); glUseProgram(0); glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND);

    char info[128];
    const char* mNames[] = { "LIT", "AO_RAW", "AO_BLUR", "NORM", "DEPTH", "POS", "DELTA" };
    sprintf(info, "Mode: %s | Bias: %.3f | Rad: %.1f", mNames[g_debugMode % 7], g_bias, g_radius);
    Alice::drawString(info, 10, 20);
    drawGrid(50);
}

void keyPress(unsigned char k, int, int) {
    if (k == 'r') setup();
    if (k == 'd') g_debugMode = (g_debugMode + 1) % 7;
    if (k == 'b') g_enableBlur = !g_enableBlur;
    if (k == '1') g_bias -= 0.05f; if (k == '2') g_bias += 0.05f;
    if (k == '3') g_radius -= 0.5f; if (k == '4') g_radius += 0.5f;
}
void mousePress(int, int, int, int) {} void mouseMotion(int, int) {}

#endif