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
float g_bias = 0.1f;
float g_radius = 12.0f;
int   g_debugMode = 2;     // Start in AO_BLUR
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
// FBO Wrapper
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
            // High precision RGBA16F for Position/Normal/AO
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
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    void bind() { glBindFramebuffer(GL_FRAMEBUFFER, id); glViewport(0, 0, w, h); glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); }
    void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }
};
FBO gBuffer; FBO ssaoFBO;

// --------------------------------------------------------------------------------
// GEOMETRY SHADER
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

// --------------------------------------------------------------------------------
// SSAO SHADER (512 Samples)
// --------------------------------------------------------------------------------
#define NUM_SAMPLES 512

class SSAOShader {
public:
    GLuint pid;
    float kernel[NUM_SAMPLES * 3]; // 512 samples

    void init() {
        uniform_real_distribution<float> rnd(0, 1); default_random_engine gen;
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            float s = (float)i / (float)NUM_SAMPLES;
            s = 0.1 + 0.9 * (s * s); // Strong cluster at origin

            float x = rnd(gen) * 2.0f - 1.0f;
            float y = rnd(gen) * 2.0f - 1.0f;
            float z = rnd(gen);

            float len = sqrt(x * x + y * y + z * z);
            x = (x / len) * s;
            y = (y / len) * s;
            z = (z / len) * s;

            kernel[i * 3 + 0] = x; kernel[i * 3 + 1] = y; kernel[i * 3 + 2] = z;
        }
        string fs = R"(
            #version 400
            out float Occ; in vec2 UV;
            uniform sampler2D gP, gN; 
            uniform vec3 k[512];  // 512 SAMPLES
            uniform mat4 P;
            uniform float rad, bias, width, height;
            
            float rand(vec2 n){return fract(sin(dot(n,vec2(12.9898,4.1414)))*43758.5453);}
            
            void main(){
                vec4 pDat = texture(gP, UV);
                if(pDat.a < 0.5) { Occ=1.0; return; } 
                vec3 pos = pDat.xyz;
                vec3 norm = normalize(texture(gN, UV).rgb);
                
                // Force Normal to face camera
                if (dot(norm, vec3(0,0,1)) < 0.0) norm = -norm;

                vec2 nUV = UV * vec2(width, height) / 4.0;
                vec3 rv = normalize(vec3(rand(nUV), rand(nUV+0.1), 0));
                vec3 T = normalize(rv - norm * dot(rv, norm));
                vec3 B = cross(norm, T);
                mat3 TBN = mat3(T, B, norm);

                float occ = 0.0;
                for(int i=0; i<512; ++i) { // 512 Loop
                    vec3 sPos = pos + (TBN * k[i]) * rad;
                    vec4 off = P * vec4(sPos, 1.0);
                    off.xyz /= off.w; off.xyz = off.xyz * 0.5 + 0.5;
                    
                    if(off.x<0||off.x>1||off.y<0||off.y>1) continue;
                    
                    float d = texture(gP, off.xy).z;
                    if (texture(gP, off.xy).a < 0.5) d = -99999.0; 

                    float range = smoothstep(0.0, 1.0, rad / abs(pos.z - d));
                    if (d >= sPos.z + bias) occ += 1.0 * range;
                }
                
                Occ = 1.0 - (occ / 512.0); // Normalize by 512
            }
        )";
        string vs = "#version 400\n layout(location=0) in vec3 P; layout(location=1) in vec2 T; out vec2 UV; void main(){UV=T; gl_Position=vec4(P,1);}";
        create(vs, fs);
    }
    void bind(GLuint p, GLuint n, Mat4& proj, int w, int h) {
        glUseProgram(pid);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, p); glUniform1i(glGetUniformLocation(pid, "gP"), 0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, n); glUniform1i(glGetUniformLocation(pid, "gN"), 1);
        glUniform3fv(glGetUniformLocation(pid, "k"), NUM_SAMPLES, kernel);
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

// --------------------------------------------------------------------------------
// BILATERAL BLUR SHADER (Edge-Preserving)
// --------------------------------------------------------------------------------
class BlurShader {
public:
    GLuint pid;
    void init() {
        string fs = R"(
            #version 400
            out vec4 C; in vec2 UV;
            uniform sampler2D sIn, gP, gN; uniform int mode; uniform bool blur; uniform float width, height;
            
            void main(){
                vec4 pC = texture(gP, UV);
                if(pC.a < 0.5) { C=vec4(0.9); return; } // Background

                float result = texture(sIn, UV).r;
                
                if(blur) {
                    vec2 ts = 1.0/vec2(width,height);
                    float totalW = 0.0;
                    float totalOcc = 0.0;
                    
                    float depthC = pC.z;
                    vec3 normC = texture(gN, UV).rgb;

                    // 4x4 Kernel
                    for(int x=-2; x<=2; x++) {
                        for(int y=-2; y<=2; y++) {
                            vec2 off = vec2(x,y) * ts;
                            
                            // Sample Neighbor
                            float occN = texture(sIn, UV + off).r;
                            float depthN = texture(gP, UV + off).z;
                            vec3 normN = texture(gN, UV + off).rgb;

                            // 1. Spatial Weight (Gaussian)
                            float w = exp(-(x*x + y*y) / 2.0);

                            // 2. Depth Weight (Prevent bleeding across gaps)
                            // If depth difference is large, weight drops to 0
                            float diffD = abs(depthC - depthN);
                            float wD = 1.0 / (1.0 + diffD * 100.0); 

                            // 3. Normal Weight (Prevent bleeding around corners)
                            float wN = max(0.0, dot(normC, normN));
                            
                            // Combined Weight
                            float weight = w * wD * wN;

                            totalOcc += occN * weight;
                            totalW += weight;
                        }
                    }
                    if (totalW > 0.0) result = totalOcc / totalW;
                }
                
                result = pow(result, 3.0); 

                // --- MODES ---
                if(mode==1) { C=vec4(vec3(texture(sIn,UV).r),1); return; } // RAW
                if(mode==2) { C=vec4(vec3(result),1); return; } // BLURRED
                if(mode==3) { C=vec4(texture(gN,UV).rgb*0.5+0.5,1); return; } // NORM
                if(mode==4) { C=vec4(vec3(clamp((-pC.z-20)/100,0,1)),1); return; } // DEPTH
                if(mode==5) { C=vec4(abs(pC.xyz)/50,1); return; } // POS
                if(mode==6) { float d = 1.0-texture(sIn,UV).r; C=vec4(d,0,0,1); return; } // DELTA

                vec3 L=normalize(vec3(0.5,0.5,1));
                float diff=max(dot(texture(gN,UV).rgb,L),0);
                C=vec4(vec3(0.3*result + 0.7*diff),1);
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

struct Cube {
    GLuint vao; int cnt;
    void init(float s) {
        float h = s * 0.5f;
        float v[] = {
            -h,-h,-h, 0,-1,0,  h,-h,-h, 0,-1,0,  h,-h, h, 0,-1,0, -h,-h, h, 0,-1,0,
             h, h,-h, 0, 1,0, -h, h,-h, 0, 1,0, -h, h, h, 0, 1,0,  h, h, h, 0, 1,0,
            -h, h,-h,-1, 0,0, -h,-h,-h,-1, 0,0, -h,-h, h,-1, 0,0, -h, h, h,-1, 0,0,
             h,-h,-h, 1, 0,0,  h, h,-h, 1, 0,0,  h, h, h, 1, 0,0,  h,-h, h, 1, 0,0,
            -h,-h,-h, 0, 0,-1, -h, h,-h, 0, 0,-1,  h, h,-h, 0, 0,-1,  h,-h,-h, 0, 0,-1,
            -h,-h, h, 0, 0, 1,  h,-h, h, 0, 0, 1,  h, h, h, 0, 0, 1, -h, h, h, 0, 0, 1
        };
        unsigned int idx[] = { 0,1,2, 0,2,3, 4,5,6, 4,6,7, 8,9,10, 8,10,11, 12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23 };
        cnt = 36;
        glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        GLuint vb, eb; glGenBuffers(1, &vb); glBindBuffer(GL_ARRAY_BUFFER, vb); glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
        glGenBuffers(1, &eb); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eb); glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, 0, 24, 0); glEnableVertexAttribArray(0); glVertexAttribPointer(1, 3, GL_FLOAT, 0, 24, (void*)12); glEnableVertexAttribArray(1);
    }
    void draw() { glBindVertexArray(vao); glDrawElements(GL_TRIANGLES, cnt, GL_UNSIGNED_INT, 0); }
};

GeomShader gs; SSAOShader ss; BlurShader bs;
Quad quad; Sphere sphere; Cube cube; Cube floorObj;

void setup() {
    gs.init(); ss.init(); bs.init(); quad.init();
    sphere.init(4.0f, 32, 32);
    cube.init(10.0f);
    floorObj.init(1.0f);
    resetCamera(); Alice::setCamera(150, 30, 45, 0, 0);
    glShadeModel(GL_SMOOTH);
    cout << "Mode: [D]. Bias: [1/2]. Rad: [3/4]." << endl;
}

void update(int v) { g_rotation += 0.01f; }

void draw() {
    int w = glutGet(GLUT_WINDOW_WIDTH); int h = glutGet(GLUT_WINDOW_HEIGHT);
    gBuffer.resize(w, h, 2); ssaoFBO.resize(w, h, 1);

    Mat4 P = perspective(60.0f, (float)w / h, 1.0f, 1000.0f);
    float mr[16];
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();
    Alice::updateCamera(); glGetFloatv(GL_MODELVIEW_MATRIX, mr);
    Mat4 V; memcpy(V.m, mr, 16 * 4);

    // --- 1. G-BUFFER ---
    glDisable(GL_BLEND);
    gBuffer.bind(); glEnable(GL_DEPTH_TEST);

    // Floor
    glPushMatrix(); glTranslatef(0, 0, -2); glScalef(120, 120, 1);
    float mf[16]; glGetFloatv(GL_MODELVIEW_MATRIX, mf);
    Mat4 Mf; memcpy(Mf.m, mf, 16 * 4); Mat4 MVPf = multiply(P, Mf);
    gs.bind(MVPf, Mf); floorObj.draw(); glPopMatrix();

    // Central Tower
    for (int k = 0; k < 3; k++) {
        glPushMatrix(); glTranslatef(0, 0, 5.0f + k * 10.0f); glRotatef(g_rotation * 20.0f * (k + 1), 0, 0, 1);
        float mt[16]; glGetFloatv(GL_MODELVIEW_MATRIX, mt);
        Mat4 Mt; memcpy(Mt.m, mt, 16 * 4); Mat4 MVPt = multiply(P, Mt);
        gs.bind(MVPt, Mt); cube.draw(); glPopMatrix();
    }

    // Grid of Spheres
    for (int x = -2; x <= 2; x++) for (int y = -2; y <= 2; y++) {
        if (x == 0 && y == 0) continue;
        glPushMatrix(); glTranslatef(x * 15.0f, y * 15.0f, 4.0f);
        float ms[16]; glGetFloatv(GL_MODELVIEW_MATRIX, ms);
        Mat4 Ms; memcpy(Ms.m, ms, 16 * 4); Mat4 MVPs = multiply(P, Ms);
        gs.bind(MVPs, Ms); sphere.draw(); glPopMatrix();
    }

    // Satellite Cubes
    for (int k = 0; k < 4; k++) {
        glPushMatrix();
        float ang = k * (PI_F / 2.0f) + g_rotation * 0.5f;
        float rad = 40.0f;
        glTranslatef(cos(ang) * rad, sin(ang) * rad, 30.0f);
        glRotatef(g_rotation * 50.0f, 1, 1, 0);
        float mc[16]; glGetFloatv(GL_MODELVIEW_MATRIX, mc);
        Mat4 Mc; memcpy(Mc.m, mc, 16 * 4); Mat4 MVPc = multiply(P, Mc);
        gs.bind(MVPc, Mc); cube.draw(); glPopMatrix();
    }

    gBuffer.unbind();

    // --- 2. SSAO ---
    ssaoFBO.bind(); glDisable(GL_DEPTH_TEST);
    ss.bind(gBuffer.textures[0], gBuffer.textures[1], P, w, h);
    quad.draw();
    ssaoFBO.unbind();

    // --- 3. BLUR/LIT ---
    glDisable(GL_BLEND);
    glClearColor(1, 1, 1, 1); glClear(GL_COLOR_BUFFER_BIT);
    bs.bind(ssaoFBO.textures[0], gBuffer.textures[0], gBuffer.textures[1], w, h);
    quad.draw();

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