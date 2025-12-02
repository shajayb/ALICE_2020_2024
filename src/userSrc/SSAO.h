#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <cmath>
#include <functional>
#include "main.h" 

// --------------------------------------------------------------------------------
// CONFIGURATION
// --------------------------------------------------------------------------------
#define MAX_SSAO_SAMPLES 1024 

// --------------------------------------------------------------------------------
// MATH & TYPES
// --------------------------------------------------------------------------------
struct vec3f {
    float x, y, z;
    vec3f(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}
};

struct mat4f { float m[16]; };

inline mat4f identity4f() { return { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }; }

inline mat4f transform4f(vec3f t, vec3f s) {
    mat4f R = identity4f();
    R.m[0] = s.x; R.m[5] = s.y; R.m[10] = s.z;
    R.m[12] = t.x; R.m[13] = t.y; R.m[14] = t.z;
    return R;
}

inline mat4f multiply(mat4f A, mat4f B) {
    mat4f R;
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            R.m[c * 4 + r] = A.m[0 * 4 + r] * B.m[c * 4 + 0] + A.m[1 * 4 + r] * B.m[c * 4 + 1] + A.m[2 * 4 + r] * B.m[c * 4 + 2] + A.m[3 * 4 + r] * B.m[c * 4 + 3];
    return R;
}

inline mat4f perspective(float fov, float aspect, float znear, float zfar) {
    float f = 1.0f / tan(fov * 0.5f * 3.14159265f / 180.0f);
    mat4f R = { 0 };
    R.m[0] = f / aspect; R.m[5] = f;
    R.m[10] = (zfar + znear) / (znear - zfar);
    R.m[11] = -1.0f;
    R.m[14] = (2.0f * zfar * znear) / (znear - zfar);
    return R;
}

// --------------------------------------------------------------------------------
// DATA CONTAINER
// --------------------------------------------------------------------------------
struct SSAOMesh {
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<unsigned int> indices;
    vec3f pos = { 0,0,0 };
    vec3f scale = { 1,1,1 };
    GLuint vao = 0; GLuint vbo[2] = { 0,0 }; GLuint ebo = 0; bool dirty = true;

    void updateGL() {
        if (!vao) glGenVertexArrays(1, &vao);
        if (!vbo[0]) glGenBuffers(2, vbo);
        if (!ebo) glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); glBufferData(GL_ARRAY_BUFFER, vertices.size() * 4, vertices.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); glBufferData(GL_ARRAY_BUFFER, normals.size() * 4, normals.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(1);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * 4, indices.data(), GL_STATIC_DRAW);
        glBindVertexArray(0); dirty = false;
    }
    void draw() {
        if (dirty || vao == 0) updateGL();
        glBindVertexArray(vao); glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0); glBindVertexArray(0);
    }
};

// --------------------------------------------------------------------------------
// INTERNAL SHADER CLASSES
// --------------------------------------------------------------------------------
class _InternalShader {
protected:
    GLuint pid;
    void create(std::string v, std::string f) {
        GLuint vs = glCreateShader(GL_VERTEX_SHADER); const char* vv = v.c_str(); glShaderSource(vs, 1, &vv, 0); glCompileShader(vs);
        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER); const char* ff = f.c_str(); glShaderSource(fs, 1, &ff, 0); glCompileShader(fs);
        pid = glCreateProgram(); glAttachShader(pid, vs); glAttachShader(pid, fs); glLinkProgram(pid);
        GLint success; glGetProgramiv(pid, GL_LINK_STATUS, &success);
        if (!success) { char info[512]; glGetProgramInfoLog(pid, 512, NULL, info); std::cout << "Shader Error: " << info << std::endl; }
    }
};

class SimpleSSAO {
private:
    struct FBO {
        GLuint id = 0; int w = 0, h = 0; std::vector<GLuint> texs;
        void resize(int _w, int _h, int n) {
            if (w == _w && h == _h) return;
            if (id) { glDeleteFramebuffers(1, &id); glDeleteTextures(texs.size(), texs.data()); texs.clear(); }
            w = _w; h = _h;
            glGenFramebuffers(1, &id); glBindFramebuffer(GL_FRAMEBUFFER, id);
            std::vector<GLenum> dbs;
            for (int i = 0; i < n; i++) {
                GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, 0);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, t, 0);
                texs.push_back(t); dbs.push_back(GL_COLOR_ATTACHMENT0 + i);
            }
            GLuint r; glGenRenderbuffers(1, &r); glBindRenderbuffer(GL_RENDERBUFFER, r);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, r);
            glDrawBuffers(dbs.size(), dbs.data()); glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        void bind() { glBindFramebuffer(GL_FRAMEBUFFER, id); glViewport(0, 0, w, h); glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); }
        void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }
    };

    struct RenderItem { SSAOMesh* mesh; mat4f modelMatrix; };
    FBO gBuffer; FBO ssaoBuffer; std::vector<RenderItem> renderQueue;

    // --- GEOMETRY SHADER ---
    class GShader : public _InternalShader {
    public:
        void init() {
            create(
                "#version 400\n layout(location=0) in vec3 P; layout(location=1) in vec3 N; uniform mat4 MVP,MV; out vec3 vP,vN; void main(){ vec4 p=MV*vec4(P,1); vP=p.xyz; vN=normalize(transpose(inverse(mat3(MV)))*N); gl_Position=MVP*vec4(P,1); }",
                "#version 400\n layout(location=0) out vec4 gP; layout(location=1) out vec4 gN; in vec3 vP,vN; void main(){ gP=vec4(vP,1); gN=vec4(normalize(vN),1); }"
            );
        }
        void bind(mat4f& MVP, mat4f& MV) { glUseProgram(pid); glUniformMatrix4fv(glGetUniformLocation(pid, "MVP"), 1, 0, MVP.m); glUniformMatrix4fv(glGetUniformLocation(pid, "MV"), 1, 0, MV.m); }
    } gs;

    class SShader : public _InternalShader {
    public:
        float kernel[MAX_SSAO_SAMPLES * 3];
        void init() {
            std::default_random_engine gen; std::uniform_real_distribution<float> rnd(0, 1);
            for (int i = 0; i < MAX_SSAO_SAMPLES; ++i) {
                float s = (float)i / (float)MAX_SSAO_SAMPLES; s = 0.1f + 0.9f * s * s;
                float x = rnd(gen) * 2 - 1; float y = rnd(gen) * 2 - 1; float z = rnd(gen);
                float l = sqrt(x * x + y * y + z * z);
                kernel[i * 3] = (x / l) * s; kernel[i * 3 + 1] = (y / l) * s; kernel[i * 3 + 2] = (z / l) * s;
            }
            create("#version 400\n layout(location=0) in vec3 P; void main(){gl_Position=vec4(P,1);}",
                R"(
                   #version 400
                   out float Occ; uniform sampler2D gP,gN; uniform vec3 k[1024]; uniform int sampleCount;
                   uniform mat4 P; uniform float rad,bias,W,H;
                   float rand(vec2 n){return fract(sin(dot(n,vec2(12.9,78.2)))*43758.5);}
                   void main(){
                       vec2 uv=gl_FragCoord.xy/vec2(W,H); vec4 pD=texture(gP,uv);
                       if(pD.a<0.5){Occ=1;return;} 
                       vec3 p=pD.xyz; vec3 n=normalize(texture(gN,uv).rgb);
                       if(dot(n,vec3(0,0,1))<0) n=-n; 
                       vec3 rv=normalize(vec3(rand(uv*W),rand(uv*H),0));
                       vec3 T=normalize(rv-n*dot(rv,n)); mat3 TBN=mat3(T,cross(n,T),n);
                       float o=0;
                       for(int i=0; i<sampleCount; i++){
                           vec3 sP=p+(TBN*k[i])*rad; vec4 off=P*vec4(sP,1); off.xy/=off.w; off.xy=off.xy*0.5+0.5;
                           if(off.x<0||off.y<0||off.x>1||off.y>1)continue;
                           float d=texture(gP,off.xy).z; 
                           if(texture(gP,off.xy).a<0.5) d=-9e9; 
                           float rng=smoothstep(0,1,rad/abs(p.z-d));
                           if(d>=sP.z+bias) o+=rng;
                       }
                       Occ=1.0-(o/float(sampleCount));
                   })");
        }
        void bind(GLuint p, GLuint n, mat4f& proj, float r, float b, int samples, int w, int h) {
            glUseProgram(pid); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, p); glUniform1i(glGetUniformLocation(pid, "gP"), 0);
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, n); glUniform1i(glGetUniformLocation(pid, "gN"), 1);
            glUniform3fv(glGetUniformLocation(pid, "k"), MAX_SSAO_SAMPLES, kernel); glUniform1i(glGetUniformLocation(pid, "sampleCount"), samples);
            glUniformMatrix4fv(glGetUniformLocation(pid, "P"), 1, 0, proj.m);
            glUniform1f(glGetUniformLocation(pid, "rad"), r); glUniform1f(glGetUniformLocation(pid, "bias"), b);
            glUniform1f(glGetUniformLocation(pid, "W"), (float)w); glUniform1f(glGetUniformLocation(pid, "H"), (float)h);
        }
    } ss;

    class BShader : public _InternalShader {
    public:
        void init() {
            create("#version 400\n layout(location=0) in vec3 P; void main(){gl_Position=vec4(P,1);}",
                R"(#version 400
            out vec4 C; uniform sampler2D sIn,gP,gN; uniform int mode; uniform bool blur; uniform float W,H;
            void main(){
                vec2 uv=gl_FragCoord.xy/vec2(W,H); vec4 pD=texture(gP,uv);
                if(pD.a<0.5){C=vec4(0.9);return;}
                float res=texture(sIn,uv).r;
                if(blur){
                    float tot=0, wTot=0; float cD=pD.z; vec3 cN=texture(gN,uv).rgb; vec2 ts=1.0/vec2(W,H);
                    for(int x=-2;x<=2;x++) for(int y=-2;y<=2;y++){
                        vec2 off=vec2(x,y)*ts; float sOcc=texture(sIn,uv+off).r;
                        float wD=1.0/(1.0+abs(cD-texture(gP,uv+off).z)*100);
                        float wN=max(0,dot(cN,texture(gN,uv+off).rgb));
                        float w=exp(-(x*x+y*y)/2.0)*wD*wN;
                        tot+=sOcc*w; wTot+=w;
                    }
                    if(wTot>0) res=tot/wTot;
                }
                res=pow(res,3.0);
                if(mode==1) C=vec4(vec3(texture(sIn,uv).r),1);
                else if(mode==2) C=vec4(vec3(res),1);
                else if(mode==3) C=vec4(texture(gN,uv).rgb*0.5+0.5,1);
                else if(mode==4) C=vec4(vec3(clamp((-pD.z-20)/100,0,1)),1);
                else if(mode==5) C=vec4(abs(pD.xyz)/50,1);
                else { 
                    vec3 L=normalize(vec3(0.5,0.5,1)); float d=max(dot(texture(gN,uv).rgb,L),0);
                    C=vec4(vec3(0.3*res + 0.7*d),1); 
                }
            })");
        }
        void bind(GLuint s, GLuint p, GLuint n, int m, bool b, int w, int h) {
            glUseProgram(pid); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, s); glUniform1i(glGetUniformLocation(pid, "sIn"), 0);
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, p); glUniform1i(glGetUniformLocation(pid, "gP"), 1);
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, n); glUniform1i(glGetUniformLocation(pid, "gN"), 2);
            glUniform1i(glGetUniformLocation(pid, "mode"), m); glUniform1i(glGetUniformLocation(pid, "blur"), b);
            glUniform1f(glGetUniformLocation(pid, "W"), (float)w); glUniform1f(glGetUniformLocation(pid, "H"), (float)h);
        }
    } bs;

    struct Quad { GLuint vao; void init() { float q[] = { -1,1,0,-1,-1,0,1,1,0,1,-1,0 }; glGenVertexArrays(1, &vao); glBindVertexArray(vao); GLuint v; glGenBuffers(1, &v); glBindBuffer(GL_ARRAY_BUFFER, v); glBufferData(GL_ARRAY_BUFFER, sizeof(q), q, GL_STATIC_DRAW); glVertexAttribPointer(0, 3, GL_FLOAT, 0, 0, 0); glEnableVertexAttribArray(0); } void draw() { glBindVertexArray(vao); glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); } } quad;

public:
    float bias = 0.1f;
    float radius = 5.0f;
    int samples = 32;
    int mode = 0;
    bool blur = true;

    void setup() { gs.init(); ss.init(); bs.init(); quad.init(); }

    void addObject(SSAOMesh* mesh, mat4f modelMatrix) { renderQueue.push_back({ mesh, modelMatrix }); }
    void addObject(SSAOMesh* mesh, vec3f pos) { renderQueue.push_back({ mesh, transform4f(pos, {1,1,1}) }); }
    void clearQueue() { renderQueue.clear(); }

    void draw() {
        // 1. Context & Projection
        int w = glutGet(GLUT_WINDOW_WIDTH);
        int h = glutGet(GLUT_WINDOW_HEIGHT);
        mat4f P = perspective(60.0f, (float)w / h, 1.0f, 1000.0f);

        // 2. View Matrix (Capture from main.h Framework)
        float vRaw[16];
        glGetFloatv(GL_MODELVIEW_MATRIX, vRaw);
        mat4f V;
        memcpy(V.m, vRaw, 16 * 4);

        // 3. Geometry Pass
        gBuffer.resize(w, h, 2); ssaoBuffer.resize(w, h, 1);
        glDisable(GL_BLEND);
        gBuffer.bind();
        glEnable(GL_DEPTH_TEST);

        for (const auto& item : renderQueue) {
            if (!item.mesh) continue;
            mat4f MV = multiply(V, item.modelMatrix);
            mat4f MVP = multiply(P, MV);
            gs.bind(MVP, MV);
            item.mesh->draw();
        }

        gBuffer.unbind();

        // 4. SSAO Pass
        if (samples > MAX_SSAO_SAMPLES) samples = MAX_SSAO_SAMPLES;
        if (samples < 1) samples = 1;

        ssaoBuffer.bind(); glDisable(GL_DEPTH_TEST);
        ss.bind(gBuffer.texs[0], gBuffer.texs[1], P, radius, bias, samples, w, h);
        quad.draw();
        ssaoBuffer.unbind();

        // 5. Lighting Pass
        glDisable(GL_BLEND); glClearColor(1, 1, 1, 1); glClear(GL_COLOR_BUFFER_BIT);
        bs.bind(ssaoBuffer.texs[0], gBuffer.texs[0], gBuffer.texs[1], mode, blur, w, h);
        quad.draw();

        glUseProgram(0); glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND);
    }
};