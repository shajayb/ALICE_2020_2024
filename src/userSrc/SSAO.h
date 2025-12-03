#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <cmath>
#include <functional>
#include "main.h" 

#define MAX_SSAO_SAMPLES 1024 

// --------------------------------------------------------------------------------
// MATH & TYPES
// --------------------------------------------------------------------------------
struct vec3f { float x, y, z; vec3f(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {} };
struct mat4f { float m[16]; };

inline mat4f identity4f() { return { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }; }

inline vec3f normalize(vec3f v) { float l = sqrt(v.x * v.x + v.y * v.y + v.z * v.z); if (l == 0) return { 0,0,0 }; return { v.x / l, v.y / l, v.z / l }; }
inline vec3f cross(vec3f a, vec3f b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
inline float dot(vec3f a, vec3f b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline mat4f transform4f(vec3f t, vec3f s) {
    mat4f R = identity4f();
    R.m[0] = s.x; R.m[5] = s.y; R.m[10] = s.z;
    R.m[12] = t.x; R.m[13] = t.y; R.m[14] = t.z;
    return R;
}

inline mat4f alignToDir(vec3f dir) {
    vec3f x = normalize(dir);
    vec3f up = { 0,0,1 };
    if (abs(dot(x, up)) > 0.99f) up = { 0,1,0 };
    vec3f y = normalize(cross(up, x));
    vec3f z = cross(x, y);
    mat4f R = identity4f();
    R.m[0] = x.x; R.m[1] = x.y; R.m[2] = x.z;
    R.m[4] = y.x; R.m[5] = y.y; R.m[6] = y.z;
    R.m[8] = z.x; R.m[9] = z.y; R.m[10] = z.z;
    return R;
}

inline mat4f multiply(mat4f A, mat4f B) {
    mat4f R;
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            R.m[c * 4 + r] = A.m[0 * 4 + r] * B.m[c * 4 + 0] + A.m[1 * 4 + r] * B.m[c * 4 + 1] + A.m[2 * 4 + r] * B.m[c * 4 + 2] + A.m[3 * 4 + r] * B.m[c * 4 + 3];
    return R;
}

// --------------------------------------------------------------------------------
// DATA CONTAINER
// --------------------------------------------------------------------------------
struct SSAOMesh {
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<unsigned int> indices;
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
        GLuint id = 0; int w = 0, h = 0; std::vector<GLuint> texs; float savedClearColor[4] = { 0,0,0,0 };
        void resize(int _w, int _h, int n) {
            if (w == _w && h == _h) return;
            if (id) { glDeleteFramebuffers(1, &id); glDeleteTextures(texs.size(), texs.data()); texs.clear(); }
            w = _w; h = _h;
            glGenFramebuffers(1, &id); glBindFramebuffer(GL_FRAMEBUFFER, id);
            std::vector<GLenum> dbs;
            for (int i = 0; i < n; i++) {
                GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, 0);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, t, 0);
                texs.push_back(t); dbs.push_back(GL_COLOR_ATTACHMENT0 + i);
            }
            GLuint r; glGenRenderbuffers(1, &r); glBindRenderbuffer(GL_RENDERBUFFER, r);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, r);
            glDrawBuffers(dbs.size(), dbs.data()); glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        void bind() { glGetFloatv(GL_COLOR_CLEAR_VALUE, savedClearColor); glBindFramebuffer(GL_FRAMEBUFFER, id); glViewport(0, 0, w, h); glClearColor(0, 0, 0, 0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); }
        void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); glClearColor(savedClearColor[0], savedClearColor[1], savedClearColor[2], savedClearColor[3]); }
    };

    struct RenderItem { SSAOMesh* mesh; mat4f modelMatrix; vec3f color; };
    FBO gBuffer; FBO ssaoBuffer; std::vector<RenderItem> renderQueue;

    // --- G-BUFFER SHADER (Pos, Norm, Color) ---
    class GShader : public _InternalShader {
    public:
        void init() {
            create(
                "#version 400\n layout(location=0) in vec3 P; layout(location=1) in vec3 N; uniform mat4 MVP,MV; out vec3 vP,vN; void main(){ vec4 p=MV*vec4(P,1); vP=p.xyz; vN=normalize(transpose(inverse(mat3(MV)))*N); gl_Position=MVP*vec4(P,1); }",
                // Output 0: Pos, 1: Norm, 2: Color
                "#version 400\n layout(location=0) out vec4 gP; layout(location=1) out vec4 gN; layout(location=2) out vec4 gC; \
                 in vec3 vP,vN; uniform vec3 objColor; void main(){ gP=vec4(vP,1); gN=vec4(normalize(vN),1); gC=vec4(objColor,1); }"
            );
        }
        void bind(mat4f& MVP, mat4f& MV, vec3f col) {
            glUseProgram(pid);
            glUniformMatrix4fv(glGetUniformLocation(pid, "MVP"), 1, 0, MVP.m);
            glUniformMatrix4fv(glGetUniformLocation(pid, "MV"), 1, 0, MV.m);
            glUniform3f(glGetUniformLocation(pid, "objColor"), col.x, col.y, col.z);
        }
    } gs;

    // --- SSAO SHADER ---
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
                   out vec4 FragData;
                   uniform sampler2D gP,gN; uniform vec3 k[1024]; uniform int sampleCount;
                   uniform mat4 P; uniform float rad,bias,W,H; uniform int mode;
                   float rand(vec2 n){return fract(sin(dot(n,vec2(12.9,78.2)))*43758.5);}
                   void main(){
                       vec2 uv=gl_FragCoord.xy/vec2(W,H); vec4 pD=texture(gP,uv);
                       if(pD.a<0.5){ FragData=vec4(1); return; } 
                       vec3 p=pD.xyz; vec3 n=normalize(texture(gN,uv).rgb);
                       if(dot(n,vec3(0,0,1))<0) n=-n; 
                       vec3 rv=normalize(vec3(rand(uv*W),rand(uv*H),0));
                       vec3 T=normalize(rv-n*dot(rv,n)); mat3 TBN=mat3(T,cross(n,T),n);
                       
                       float occ=0, c_red=0, c_grn=0, c_blu=0;
                       for(int i=0; i<sampleCount; i++){
                           vec3 sP=p+(TBN*k[i])*rad; vec4 off=P*vec4(sP,1); off.xy/=off.w; off.xy=off.xy*0.5+0.5;
                           if(off.x<0||off.y<0||off.x>1||off.y>1) continue;
                           vec4 sVal=texture(gP,off.xy);
                           float d=sVal.z; bool isSky=sVal.a<0.5;
                           if(isSky) { d=-9e9; c_blu+=1.0; } 
                           float rng=smoothstep(0,1,rad/abs(p.z-d));
                           if(d>=sP.z+bias) { occ+=1.0*rng; c_red+=1.0*rng; } 
                           else if(!isSky) { c_grn+=1.0; }
                       }
                       float finalOcc = 1.0 - (occ / float(sampleCount));
                       if(mode == 7) FragData = vec4(c_red/sampleCount, c_grn/sampleCount, c_blu/sampleCount, 1.0);
                       else FragData = vec4(finalOcc);
                   })");
        }
        void bind(GLuint p, GLuint n, mat4f& proj, float r, float b, int samples, int m, int w, int h) {
            glUseProgram(pid); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, p); glUniform1i(glGetUniformLocation(pid, "gP"), 0);
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, n); glUniform1i(glGetUniformLocation(pid, "gN"), 1);
            glUniform3fv(glGetUniformLocation(pid, "k"), MAX_SSAO_SAMPLES, kernel);
            glUniform1i(glGetUniformLocation(pid, "sampleCount"), samples);
            glUniformMatrix4fv(glGetUniformLocation(pid, "P"), 1, 0, proj.m);
            glUniform1f(glGetUniformLocation(pid, "rad"), r); glUniform1f(glGetUniformLocation(pid, "bias"), b);
            glUniform1i(glGetUniformLocation(pid, "mode"), m);
            glUniform1f(glGetUniformLocation(pid, "W"), (float)w); glUniform1f(glGetUniformLocation(pid, "H"), (float)h);
        }
    } ss;

    class BShader : public _InternalShader {
    public:
        void init() {
            create("#version 400\n layout(location=0) in vec3 P; void main(){gl_Position=vec4(P,1);}",
                R"(#version 400
            out vec4 C; uniform sampler2D sIn,gP,gN,gAlbedo; uniform int mode; uniform bool blur; uniform float W,H;
            void main(){
                vec4 pD=texture(gP,gl_FragCoord.xy/vec2(W,H));
                if(pD.a < 0.5) discard; 
                vec2 uv = gl_FragCoord.xy/vec2(W,H);
                if (mode == 7) { C = texture(sIn, uv); return; }
                float res=texture(sIn,uv).r;
                if(blur){
                    vec2 ts = 1.0/vec2(W,H);
                    float tot=0, wTot=0; float cD=pD.z; vec3 cN=texture(gN,uv).rgb; 
                    for(int x=-2;x<=2;x++) for(int y=-2;y<=2;y++){
                        vec2 off=vec2(x,y)*ts; float sOcc=texture(sIn,uv+off).r;
                        float wD=1.0/(1.0+abs(cD-texture(gP,uv+off).z)*100);
                        float wN=max(0,dot(cN,texture(gN,uv+off).rgb));
                        float w=exp(-(x*x+y*y)/2.0)*wD*wN;
                        tot+=sOcc*w; wTot+=w;
                    }
                    if(wTot>0) res=tot/wTot;
                }
                
                // Pronounced AO Curve
                res = pow(res, 4.0); // Was 3.0
                
                // --- READ ALBEDO ---
                vec3 albedo = texture(gAlbedo, uv).rgb;

                if(mode==1) C=vec4(vec3(texture(sIn,uv).r),1);
                else if(mode==2) C=vec4(vec3(res),1);
                else if(mode==3) C=vec4(texture(gN,uv).rgb*0.5+0.5,1);
                else if(mode==4) C=vec4(vec3(clamp((-pD.z-20)/100,0,1)),1);
                else if(mode==5) C=vec4(abs(pD.xyz)/50,1);
                else { 
                    vec3 L=normalize(vec3(0.5,0.5,1)); float d=max(dot(texture(gN,uv).rgb,L),0);
                    
                    // Stronger Ambient Occlusion Effect
                    vec3 ambient = albedo * 0.4; 
                    vec3 diffuse = albedo * 0.6 * d;

                    // Apply AO to both ambient and diffuse for a 'dirtier', punchier look
                    C = vec4( (ambient + diffuse) * res, 1.0); 
                }
            })");
        }
        void bind(GLuint s, GLuint p, GLuint n, GLuint c, int m, bool b, int w, int h) {
            glUseProgram(pid);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, s); glUniform1i(glGetUniformLocation(pid, "sIn"), 0);
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, p); glUniform1i(glGetUniformLocation(pid, "gP"), 1);
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, n); glUniform1i(glGetUniformLocation(pid, "gN"), 2);
            glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, c); glUniform1i(glGetUniformLocation(pid, "gAlbedo"), 3);
            glUniform1i(glGetUniformLocation(pid, "mode"), m); glUniform1i(glGetUniformLocation(pid, "blur"), b);
            glUniform1f(glGetUniformLocation(pid, "W"), (float)w); glUniform1f(glGetUniformLocation(pid, "H"), (float)h);
        }
    } bs;

    struct Quad { GLuint vao; void init() { float q[] = { -1,1,0,-1,-1,0,1,1,0,1,-1,0 }; glGenVertexArrays(1, &vao); glBindVertexArray(vao); GLuint v; glGenBuffers(1, &v); glBindBuffer(GL_ARRAY_BUFFER, v); glBufferData(GL_ARRAY_BUFFER, sizeof(q), q, GL_STATIC_DRAW); glVertexAttribPointer(0, 3, GL_FLOAT, 0, 0, 0); glEnableVertexAttribArray(0); } void draw() { glBindVertexArray(vao); glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); } } quad;

public:
    float bias = 0.1f;
    double radius = 5.0f;
    int samples = 32;
    int mode = 0;
    bool blur = true;

    void setup() { gs.init(); ss.init(); bs.init(); quad.init(); }

    void addObject(SSAOMesh* mesh, mat4f modelMatrix, vec3f color = { 1,1,1 }) { renderQueue.push_back({ mesh, modelMatrix, color }); }
    void addObject(SSAOMesh* mesh, vec3f pos, vec3f color = { 1,1,1 }) { renderQueue.push_back({ mesh, transform4f(pos, {1,1,1}), color }); }
    void clearQueue() { renderQueue.clear(); }
    int getObjectCount() const { return (int)renderQueue.size(); }

    void draw() {
        int w = glutGet(GLUT_WINDOW_WIDTH); int h = glutGet(GLUT_WINDOW_HEIGHT);
        float pRaw[16]; glGetFloatv(GL_PROJECTION_MATRIX, pRaw); mat4f P; memcpy(P.m, pRaw, 16 * 4);
        float vRaw[16]; glGetFloatv(GL_MODELVIEW_MATRIX, vRaw); mat4f V; memcpy(V.m, vRaw, 16 * 4);

        // Resize for 3 Attachments
        gBuffer.resize(w, h, 3); ssaoBuffer.resize(w, h, 1);
        glDisable(GL_BLEND); gBuffer.bind(); glEnable(GL_DEPTH_TEST);

        for (const auto& item : renderQueue) {
            if (!item.mesh) continue;
            mat4f MV = multiply(V, item.modelMatrix);
            mat4f MVP = multiply(P, MV);
            gs.bind(MVP, MV, item.color);
            item.mesh->draw();
        }
        gBuffer.unbind();

        if (samples > MAX_SSAO_SAMPLES) samples = MAX_SSAO_SAMPLES;
        if (samples < 1) samples = 1;

        ssaoBuffer.bind(); glDisable(GL_DEPTH_TEST);
        ss.bind(gBuffer.texs[0], gBuffer.texs[1], P, radius, bias, samples, mode, w, h);
        quad.draw(); ssaoBuffer.unbind();

        glDisable(GL_BLEND);
        // Bind Albedo Texture (Index 2)
        bs.bind(ssaoBuffer.texs[0], gBuffer.texs[0], gBuffer.texs[1], gBuffer.texs[2], mode, blur, w, h);
        quad.draw();

        glUseProgram(0); glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND);
    }
};