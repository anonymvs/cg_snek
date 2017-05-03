//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Hegedus Daniel
// Neptun : YBY8BK
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}


/*
// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";
*/


// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }

	mat4 tranpose() {
		mat4 ret;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				ret.m[j][i] = m[i][j];
			}
		}
		return ret;
	}

	void SetUniform(unsigned shaderProg, const char * name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniformMatrix4fv(loc, 1, GL_TRUE, &m[0][0]);
	}
};

//transformation matrices
mat4 Translate(float tx, float ty, float tz) {
	return mat4(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		tx, ty, tz, 1);
}
mat4 Rotate(float angle, float wx, float wy, float wz) {
	float cosa = cosf(angle);
	float sina = sinf(angle);
	float X = wx * wx;
	float Y = wy * wy;
	float Z = wz * wz;
	mat4 ret = mat4(
		cosa + X * (1 - cosa), wx * wy * (1 - cosa) - wz * sina, wy * sina + wx * wz * (1 - cosa), 0,
		wz * sina + wx * wy * (1 - cosa), cosa + Y * (1 - cosa), -1 * wx * sina + wy * wz * (1 - cosa), 0,
		-1 * wy * sina + wx * wz * (1 - cosa), wx * sina + wy * wz * (1 - cosa), cosa + Z * (1 - cosa), 0,
		0, 0, 0, 1
	);
	return ret.tranpose();
}
mat4 Scale(float sx, float sy, float sz) {
	return mat4(
		sx, 0, 0, 0,
		0, sy, 0, 0,
		0, 0, sz, 0,
		0, 0, 0, 1
	);
}

// vec3 class in world coordinates
struct vec3 {
public:
	float x, y, z;

	//vec3() {}

	vec3(float argx = 0, float argy = 0, float argz = 0) {
		x = argx;
		y = argy;
		z = argz;
	}

	vec3 operator+(const vec3 arg) {
		vec3 v(0, 0, 0);
		v.x = this->x + arg.x;
		v.y = this->y + arg.y;
		v.z = this->z + arg.z;
		return v;
	}
	vec3 operator-(const vec3 arg) {
		vec3 v(0, 0, 0);
		v.x = this->x - arg.x;
		v.y = this->y - arg.y;
		v.z = this->z - arg.z;
		return v;
	}
	vec3 operator/(const float arg) {
		vec3 v(0, 0, 0);
		v.x = this->x / arg;
		v.y = this->y / arg;
		v.z = this->z / arg;
		return v;
	}
	vec3 operator*(const float arg) {
		vec3 v(0, 0, 0);
		v.x = this->x * arg;
		v.y = this->y * arg;
		v.z = this->z * arg;
		return v;
	}
	vec3& operator=(const vec3 arg) {
		this->x = arg.x;
		this->y = arg.y;
		this->z = arg.z;
		return *this;
	}

	vec3& norm() {
		return *this = *this * (1 / sqrtf(x * x + y * y + z * z));
	}

	float length() {
		return sqrtf(x * x + y * y + z * z);
	}

	vec3 cross(const vec3 arg) {
		return vec3(y*arg.z - z*arg.y, z*arg.x - x*arg.z, x*arg.y - y * arg.x);
	}

	void SetUniform3f(unsigned shaderProg, const char * name) {
		int loc = glGetUniformLocation(shaderProg, name);
		float v[3];
		v[0] = x;
		v[1] = y;
		v[2] = z;
		glUniform3fv(loc, 1, &v[0]);
	}

	void SetUniform4f(unsigned shaderProg, char* name) {
		int loc = glGetUniformLocation(shaderProg, name);
		float v[3];
		v[0] = x;
		v[1] = y;
		v[2] = z;
		glUniform4fv(loc, 1, &v[0]);
	}

};

// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4(vec3 arg) {
		v[0] = arg.x; v[1] = arg.y; v[2] = arg.z; v[3] = 1;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	vec4 operator+(const vec4 arg) {
		vec4 ret;
		ret.v[0] = this->v[0] + arg.v[0];
		ret.v[1] = this->v[1] + arg.v[1];
		ret.v[2] = this->v[2] + arg.v[2];
		return ret;
	}
	vec4 operator-(const vec4 arg) {
		vec4 ret;
		ret.v[0] = this->v[0] - arg.v[0];
		ret.v[1] = this->v[1] - arg.v[1];
		ret.v[2] = this->v[2] - arg.v[2];
		return ret;
	}
	vec4 operator/(const float arg) {
		vec4 ret;
		ret.v[0] = this->v[0] / arg;
		ret.v[1] = this->v[1] / arg;
		ret.v[2] = this->v[2] / arg;
		return ret;
	}
	vec4 operator*(const float arg) {
		vec4 ret;
		ret.v[0] = this->v[0] * arg;
		ret.v[1] = this->v[1] * arg;
		ret.v[2] = this->v[2] * arg;
		return ret;
	}
	vec4 operator*(const vec4 arg) {
		return vec4(v[0] * arg.v[0], v[1] * arg.v[1], v[2] * arg.v[2], 1);
	}
	vec4& operator=(const vec4 arg) {
		this->v[0] = arg.v[0];
		this->v[1] = arg.v[1];
		this->v[2] = arg.v[2];
		return *this;
	}

	float dot(vec4 b) {
		return v[0] * b.v[0] + v[1] * b.v[1] + v[2] * b.v[2];
	}

	vec4& norm() {
		return *this = *this * (1 / sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
	}

	float length() {
		return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}

	void SetUniform(unsigned shaderProg, const char * name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniform4fv(loc, 4, &v[0]);
	}
};

// 3D camera
struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		//Animate(0);
	}

	void Create(vec3 awEye, vec3 awLookat, vec3 awVup, float afov, float aasp, float afp, float abp) {
		wEye = awEye;
		wLookat = awLookat;
		wVup = awVup;
		fov = afov;
		asp = aasp;
		fp = afp;
		bp = abp;
	}

	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = (wEye - wLookat).norm();
		vec3 u = wVup.cross(w).norm();
		vec3 v = w.cross(u);
		return Translate(-wEye.x, -wEye.y, -wEye.z) *
			mat4(u.x, v.x, w.x, 0,
				u.y, v.y, w.y, 0,
				u.z, v.z, w.z, 0,
				0, 0, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		float sy = 1 / tanf(fov / 2);
		return mat4(sy / asp, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp*bp / (bp - fp), 0
		);
	}
};

struct Light {
	vec3 La, Le;
	vec3 wLightPos;

	Light() {

	}

	void Create(vec3 la, vec3 le, vec3 wLightPos) {
		La = la;
		Le = le;
		this->wLightPos = wLightPos;
	}
};

struct Material
{
	vec3 kd, ks, ka;
	float  shininess;

	Material() {

	}

	Material(vec3 kd, vec3 ks, vec3 ka, float shininess) {
		this->kd = kd;
		this->ks = ks;
		this->ka = ka;
		this->shininess = shininess;
	}

	void Create(vec3 kd, vec3 ks, vec3 ka, float shininess) {
		this->kd = kd;
		this->ks = ks;
		this->ka = ka;
		this->shininess = shininess;
	}
	
	Material& operator=(const Material arg) {
		this->kd = arg.kd;
		this->ks = arg.ks;
		this->shininess = arg.shininess;
		return *this;
	}
};

struct Texture {
	unsigned int textureId;
	Texture() {

	}

	void Create() {
		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId); //binding
												 //float image[width*height*3] = LoadImage(fname, width, height);
		int height = 10;
		int width = 10;
		float image[10 * 10 * 3];
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				image[i * height + width] = 0.4f;
				image[i * height + width + 1] = 0.4f;
				image[i * height + width + 2] = 0.4f;
			}
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, image);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  // old
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	// new
	}

	Texture operator=(const Texture& arg) {
		this->textureId = arg.textureId;
		return *this;
	}
};

struct RenderState {
	mat4 M, V, P, Minv;
	Material material;
	Texture texture;
	Light light;
	vec3 wEye;

	RenderState() {

	}
};


unsigned int shaderProg;
unsigned int shaderProg_snek_body;
unsigned int shaderProg_snek_head;
// handle of the shader program
struct Shader {
	
	void Create(const char *vsSrc, const char *vsAttrNames[],
		const char *fsSrc, const char *fsOutputName, int n) {

		unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs, 1, &vsSrc, NULL);
		glCompileShader(vs);
		unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, &fsSrc, NULL);
		glCompileShader(fs);
		checkShader(vs, "Vertex Shader Error: ");
		checkShader(fs, "Fragment Shader Error: ");
		shaderProg = glCreateProgram();
		glAttachShader(shaderProg, vs);
		glAttachShader(shaderProg, fs);

		for (int i = 0; i < n; i++) {
			glBindAttribLocation(shaderProg, i, vsAttrNames[i]);
		}
		glBindFragDataLocation(shaderProg, 0, fsOutputName);
		glLinkProgram(shaderProg);
		checkLinking(shaderProg);
	}

	virtual void Bind(RenderState& state) {
		glUseProgram(shaderProg);
	}
};

class Phong : public Shader {
	const char* vsSrc = R"(
		#version 130
	    precision highp float;

		uniform mat4 MVP, M, Minv;		// MVP, 	Model, Model-Inverse
		uniform vec4 wLiPos;					// pos of light source
		uniform vec3 wEye;						// pos of eye

		in vec3 vtxPos; 							// pos in modeling space
		in vec3 vtxNorm;							// normal in modeling space
		in vec2 vtxUV;					//for texture

		out vec3 wNormal;							// normal in world space
		out vec3 wView;								// view in world space
		out vec3 wLight;							// light dir in world space
		out vec2 texcoord;				//for texture

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;  // to NDC
			texcoord = vtxUV;
			vec4 wPos = vec4(vtxPos, 1) * M;
			wLight = wLiPos.xyz - wPos.xyz;
			wView = wEye * wPos.w - wPos.xyz;
			wNormal = (Minv * vec4(vtxNorm, 1)).xyz;
		}
	)";

	const char* fsSrc = R"(
		#version 130
	    precision highp float;

		uniform vec3 kd, ks, ka; 			// diffuse, specular, ambient ref
		uniform vec3 La, Le; 					// ambient and point source rad
		uniform float shine; 					// shininess for specular ref
		uniform sampler2D samplerUnit;		//for texture

		in vec3 wNormal; 							// interpolated world sp normal
		in vec3 wView; 								// interpolated world sp view
		in vec3 wLight;								// interpolated world sp illum dir
		in vec2 texcoord;		//for texture
		out vec4 fragmentColor; 			// output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			vec3 L = normalize(wLight);
			vec3 H = normalize(L + V);
			float cost = max(dot(N,L), 0);
			float cosd = max(dot(N,H), 0);
			vec3 color = ka * La + (kd * cost + ks * pow(cosd, shine)) * Le;
			//fragmentColor = vec4(vec3(1,1,1) * cost, 1);
			fragmentColor = vec4(color, 1.0f);
			//fragmentColor = vec4(1,1,1,1);
			//fragmentColor = texture(samplerUnit, texcoord);
			//fragmentColor = texture(samplerUnit, texcoord) * color;
		}
	)";
public:
	Phong() {

	}

	void Init() {
		static const char* vsAttrNames[] = { "vtxPos" , "vtxNorm" , "vtxUV"};
		Create(vsSrc, vsAttrNames, fsSrc, "fragmentColor", 3);
	}

	void Bind(RenderState& state) {
		glUseProgram(shaderProg);
		mat4 MVP = state.M * state.V * state.P;
		MVP.SetUniform(shaderProg, "MVP");
		state.M.SetUniform(shaderProg, "M");
		state.Minv.SetUniform(shaderProg, "Minv");
		state.light.wLightPos.SetUniform4f(shaderProg, "wLiPos");
		state.wEye.SetUniform3f(shaderProg, "wEye");
		state.material.kd.SetUniform3f(shaderProg, "kd");
		state.material.ks.SetUniform3f(shaderProg, "ks");
		state.material.ka.SetUniform3f(shaderProg, "ka");
		state.light.La.SetUniform3f(shaderProg, "La");
		state.light.Le.SetUniform3f(shaderProg, "Le");
		int loc = glGetUniformLocation(shaderProg, "shine");
		glUniform1f(loc, state.material.shininess);


	}
};


//unsigned int nVtx;
//unsigned int nVtx_snek_body;

class Geometry {
public:
	unsigned int vao;
	//unsigned int nVtx;
	unsigned int nVtx;

	Geometry() {

	}

	void Create(int a) { // a = 0 head; a = 1 body
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		
	}

	void Draw(int a ) {
		//glBindVertexArray(vao);
		/*
		if (a == 0) {
			glBindVertexArray(vao);
			glDrawArrays(GL_TRIANGLES, 0, nVtx);
		}
		if (a == 1) {
			glBindVertexArray(vao);
			glDrawArrays(GL_TRIANGLES, nVtx, nVtx_snek_body);
		}
		*/

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, nVtx);
	}
};

struct VertexData {
	vec3 position, normal;
	float u, v;

	VertexData() {

	}
};

struct ParamSurface : public Geometry {
	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N, int M) {
		Geometry::Create(0);

		nVtx = N * M * 6;
		
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		
		VertexData *vtxData = new VertexData[nVtx];
		VertexData *pVtx = vtxData;

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				*pVtx++ = GenVertexData((float)i / N, (float)j / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
				*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)(j + 1) / M);
				*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
			}
		}
		//printf("nVtx: %d\n", nVtx);
		int stride = sizeof(VertexData);
		int sVec3 = sizeof(vec3);
		glBufferData(GL_ARRAY_BUFFER, nVtx * stride, vtxData, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0); // AttribArray 0 = POSITION
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
		glEnableVertexAttribArray(1); // AttribArray 1 = NORMAL
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)sVec3);
		glEnableVertexAttribArray(2); // AttribArray 2 = UV
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(2 * sVec3));
	}
};


///MATERIALS RIGHT HERERERERERERERERERR
Material material;
Material material_snek_body;


class Object {
protected:
	Shader* shader;
	Material* material1;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, pos, rotAxis;
	float rotAngle;
public:
	Object() {

	}

	void Create(Geometry *g, Material *m, Texture *t, Shader* shader) {
		geometry = g;
		material1 = m;
		texture = t;
		this->shader = shader;
	}

	virtual void Draw(RenderState state) = 0;

	virtual void Animate(float dt) = 0;
};

class Head : public Object {
public:
	Head() {

	}

	void Create(Geometry *g, Material *m, Texture *t, Shader *shader, vec3 scale, vec3 pos, vec3 rotAxis, float rotAngle) {
		Object::Create(g, m, t, shader);
		this->scale = scale;
		this->pos = pos;
		this->rotAxis = rotAxis;
		this->rotAngle = rotAngle;
	}

	void Draw(RenderState state) {
		state.M = Scale(scale.x, scale.y, scale.z) *
			Rotate(rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
			Translate(pos.x, pos.y, pos.z);
		state.Minv = Translate(-pos.x, -pos.y, -pos.z) *
			Rotate(-rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
			Scale(1 / scale.x, 1 / scale.y, 1 / scale.z);
		state.material = material;
		state.texture = *texture;
		shader->Bind(state);

		//int samplerUnit = 0;
		int location = glGetUniformLocation(shaderProg, "samplerUnit");
		//glUniform1i(location, samplerUnit);
		glUniform1i(location, GL_TEXTURE0);
		//glActiveTexture(GL_TEXTURE0 + samplerUnit);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture->textureId);

		geometry->Draw(0);
	}
	bool b = true;
	void Animate(float dt) {
		pos = vec3(cosf(dt) * 5.0f, cosf(dt) * 5.0f, 0);
		//scale = vec3(1.0f, 0.5f, 1.0f);
		//scale = vec3(1, 1, 1);
		/*
		if (b) {
			rotAngle += 0.001;
			if (rotAngle > M_PI * 2) b = false;
		}
		else {
			//rotAngle -= 0.001;
			if (rotAngle < 0.0f) b = true;
		}
		*/
		rotAngle = cosf(dt);
	}
};

class Body : public Object {
public:
	Body() {

	}

	void Create(Geometry *g, Material *m, Texture *t, Shader *shader, vec3 scale, vec3 pos, vec3 rotAxis, float rotAngle) {
		Object::Create(g, m, t, shader);
		this->scale = scale;
		this->pos = pos;
		this->rotAxis = rotAxis;
		this->rotAngle = rotAngle;
	}

	void Draw(RenderState state) {
		state.M = Scale(scale.x, scale.y, scale.z) *
			Rotate(rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
			Translate(pos.x, pos.y, pos.z);
		state.Minv = Translate(-pos.x, -pos.y, -pos.z) *
			Rotate(-rotAngle, rotAxis.x, rotAxis.y, rotAxis.z) *
			Scale(1 / scale.x, 1 / scale.y, 1 / scale.z);
		state.material = material_snek_body;
		state.texture = *texture;
		shader->Bind(state);

		//int samplerUnit = 0;
		int location = glGetUniformLocation(shaderProg, "samplerUnit");
		//glUniform1i(location, samplerUnit);
		glUniform1i(location, GL_TEXTURE0);
		//glActiveTexture(GL_TEXTURE0 + samplerUnit);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture->textureId);

		geometry->Draw(1);
	}
	void Animate(float dt) {
		pos = vec3(cosf(dt) * 5.0f, cosf(dt) * 5.0f, 0);
	}

};


//Geometries
class Sphere : public ParamSurface {
	vec3 center;
	float radius;
public:
	Sphere()  {

	}

	void Create(vec3 c, float r) {
		center = c;
		radius = r;
		ParamSurface::Create(32, 32); // tesselation level
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(cos(u * 2 * M_PI) * sin(v*M_PI),
			sin(u * 2 * M_PI) * sin(v*M_PI),
			cos(v*M_PI));
		vd.position = vd.normal * radius + center;
		vd.u = u;
		vd.v = v;
		return vd;
	}
};

class ControlPoint {
public:
	vec3 pos;
	float t;
	vec3 vv;

	ControlPoint() {}

	ControlPoint(float argt, float x, float y, float z) {
		pos = vec3(x, y, z);
		t = argt;
		vv = vec3(0, 0, 0);
	}

	ControlPoint& operator=(ControlPoint arg) {
		this->pos = arg.pos;
		this->t = arg.t;
		this->vv = arg.vv;
		return *this;
	}

};

class CatmullRom : Geometry{
	ControlPoint cps[5];
	int n;

	std::vector<vec3> spine_vertices;

public:
	CatmullRom() {
		n = 5;
		cps[0] = ControlPoint(0.0f, 0.0f, 8.0f, 0.0f);
		cps[1] = ControlPoint(0.0f, 0.0f, 6.0f, 0.0f);
		cps[2] = ControlPoint(5.0f, 5.0f, 4.0f, 0.0f);
		cps[3] = ControlPoint(0.0f, 0.0f, 2.0f, 0.0f);
		cps[4] = ControlPoint(0.0f, 0.0f, 0.0f, 0.0f);
	}

	void Create() {
		Geometry::Create(1);
		
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		//setting up time values
		for (int i = 0; i < n; i++) {
			if (i == 0) cps[i].t = 0.0f;
			else cps[i].t = getT(cps[i - 1].t, cps[i - 1].pos, cps[i].pos);
		}
		//setting up velocity vector values
		for (int i = 0; i < n-1; i++) {
			cps[i].vv =(( ( (cps[i+1].pos - cps[i].pos) /
						    (cps[i+1].t - cps[i].t) ) 
									+
					      ( (cps[i].pos - cps[i-1].pos) /
					        (cps[i].t - cps[i-1].t) )     ) 
															* 0.9f );
		}

		float dt = 0.25;
		for(float i = cps[0].t; i <= cps[n-1].t; i += dt ) {
			vec3 h = r(i);
			spine_vertices.push_back(h);
		}

		std::vector<std::vector<VertexData>> vtxMap;
		float r = 1.0f;
		for (int a = 0; a < spine_vertices.size(); a++) {
			std::vector<vec3> points;
			vec3 o(0, 0, 0);
			float rad = 0;
			for (int i = 0; i < 360; i++) {
				float x = o.x + r * cosf(rad);
				float z = o.z + r * sinf(rad);
				points.push_back(vec3(x, 0, z));
				rad++;
			}
			vec3 current = spine_vertices[a];
			vec3 dir;
			if (a == spine_vertices.size() - 1)
				dir = current - spine_vertices[a - 1];
			else
				dir = spine_vertices[a + 1] - current;
			dir = dir.norm();
			vec3 i(1, 0, 0);
			vec3 j(0, 1, 0);
			vec3 k(0, 0, 1);
			float xAngle = acosf((dir.x * i.x + dir.y * i.y + dir.z * i.z) / dir.length() * i.length());
			float yAngle = acosf((dir.x * j.x + dir.y * j.y + dir.z * j.z) / dir.length() * j.length());
			float zAngle = acosf((dir.x * k.x + dir.y * k.y + dir.z * k.z) / dir.length() * k.length());
			
			std::vector<VertexData> vtxDataVector;
			for (int n = 0; n < 360; n++) {
				vec4 tmp = vec4(points[n]) * Rotate(xAngle, i.x, i.y, i.z) * Rotate(yAngle, j.x, j.y, j.z) * Rotate(zAngle, k.x, k.y, k.z) * Translate(current.x, current.y, current.z);
				vec3 pos(tmp.v[0], tmp.v[1], tmp.v[2]);
				VertexData vd;
				vd.position = pos;
				vd.normal = pos - current;
				vd.u = a / spine_vertices.size();
				vd.v = n / 360;
				vtxDataVector.push_back(vd);
			}
			vtxMap.push_back(vtxDataVector);
		}

		nVtx = spine_vertices.size() * 360 * 6;
		VertexData *vtxData = new VertexData[nVtx];
		VertexData *pVtx = vtxData;
		unsigned int cnt = 0;
		for (int i = 0; i < 360-1; i++) {
			for (int j = 0; j < spine_vertices.size()-1; j++) {
				*pVtx++ = vtxMap[j][i];
				*pVtx++ = vtxMap[j][i+1];
				*pVtx++ = vtxMap[j+1][i];
				*pVtx++ = vtxMap[j][i+1];
				*pVtx++ = vtxMap[j+1][i+1];
				*pVtx++ = vtxMap[j+1][i];
				cnt+=6;
			}
		}

		int stride = sizeof(VertexData);
		int sVec3 = sizeof(vec3);
		glBufferData(GL_ARRAY_BUFFER, nVtx * stride, vtxData, GL_STATIC_DRAW);
		//printf("cnt: %d\n", cnt);

		glEnableVertexAttribArray(0);  // AttribArray 0 = POSITION
		glEnableVertexAttribArray(1);  // AttribArray 1 = NORMAL
		glEnableVertexAttribArray(2);  // AttribArray 2 = UV
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)sVec3);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(2 * sVec3));
	}

	float getT(float t, vec3 p0, vec3 p1) {
		float a = powf(p1.x - p0.x, 2.0f) + powf(p1.y - p0.y, 2.0f) + powf(p1.z - p0.z, 2.0f);
		float b = powf(a, 0.5f);
		float c = powf(b, 0.5f);
		return (c + t);
	}

	vec3 hermite(float t, ControlPoint cV, ControlPoint nV) {
		vec3 a0 = cV.pos;
		vec3 a1 = cV.vv;
		vec3 a2 = (((nV.pos - cV.pos)*3) / powf((nV.t - cV.t), 2.0)) - ((nV.vv + (cV.vv * 2)) / (nV.t - cV.t));
		vec3 a3 = (((cV.pos - nV.pos)*2) / powf((nV.t - cV.t), 3.0)) + ((nV.vv + cV.vv) / powf((nV.t - cV.t), 2.0));

		vec3 coord = a3 * powf((t - cV.t), 3.0) + a2 * powf((t - cV.t), 2.0) + a1 * (t - cV.t) + a0;

		return coord;
	}

	vec3 r(float t) {
		for(int i = 0; i <= n; i++) {
			if( cps[i].t <= t && t <= cps[i+1].t ) {
				vec3 h = hermite(t, cps[i], cps[i+1]);
				return h;
			}
		}
		return vec3(0,0);
	}

	ControlPoint* getCps() {
		return cps;
	}

	int getSize() {
		return n;
	}
};

class Scene {
	Camera camera;
	std::vector<Object*> objects;
	Light light;
	RenderState state;
public:
	Scene() {

	}

	void Create(Camera c, Light l, RenderState state) {
		camera = c;
		light = l;
		this->state = state;
	}

	void Render() {
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.light = light;
		for (Object * obj : objects) 
			obj->Draw(state);
	}

	void AddObject(Object* o) {
		objects.push_back(o);
	}

	void Animate(float dt) {
		for (Object * obj : objects) obj->Animate(dt);
	}
};

// The virtual world: collection of objects
Camera camera;
Scene scene;
Phong phong_snek_body;
Phong phong_snek_hed;
RenderState rs;
RenderState rs_snek_body;
Texture texture;
Texture texture_snek_body;
Head head;
Body body;
Light light;
CatmullRom snek_body;
Sphere sp;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glEnable(GL_DEPTH_TEST); // z-buffer is on
	glDisable(GL_CULL_FACE); // backface culling is off
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	// Create objects by setting up their vertex data on the GPU
	///CAMERA CREATION
	vec3 wEye(0.0f, 0.0f, 15.0f);
	vec3 wLookat(0.0f, 0.0f, 7.0f);
	vec3 wVup(0.0f, 1.0f, 0.0f); //paralell with Y axis
	float fov = 45;
	float asp = 1; //picture ratio, and is one because 600x600
	float fp = 0.1; //from this will be rendered in front of camera
	float bp = 1000; //distance it will be rendered
	camera.Create(wEye, wLookat, wVup, fov, asp, fp, bp);

	///PHONG CREATION
	phong_snek_hed.Init();
	//phong_snek_body.Init();

	///SPHERE CREATION
	//geometry
	vec3 pos(0.0f, 0.0f, 0.0f);
	float r = 3.0f;
	
	sp.Create(pos, r);

	
	snek_body.Create();

	//material
	vec3 kd(0.0f, 0.0f, 1.0f);
	vec3 ks(0.0f, 1.0f, 1.0f);
	vec3 ka(0.5f, 0.5f, 0.5f);
	float shininess = 50.0f;
	material.Create(kd, ks, ka, shininess);

	kd = vec3(0.0f, 1.0f, 0.0f);
	ks = vec3(0.0f, 1.0f, 1.0f);
	ka = vec3(0.2f, 0.2f, 0.2f);
	shininess = 50.0f;
	material_snek_body.Create(kd, ks, ka, shininess);

	//texture
	Texture t;
	t.Create();

	texture_snek_body.Create();
	

	///OBJECT CREATION
	//head
	vec3 scale = vec3(0.6f, 0.5f, 1.0f);
	pos = vec3(0, 0, 0);
	vec3 rotAxis = vec3(0, 1, 0);
	float rotAngle = 0;
	head.Create((Geometry*)&sp, &material, &t, (Shader*)&phong_snek_hed, scale, pos, rotAxis, rotAngle);

	//body
	vec3 scale_body = vec3(1, 1, 1);
	pos = vec3(0, 0, 0);
	body.Create((Geometry*)&snek_body, &material_snek_body, &texture_snek_body, (Shader*)&phong_snek_hed, scale_body, pos, rotAxis, rotAngle);

	///LIGHT CREATION
	vec3 La(0.2f, 0.2f, 0.2f);
	vec3 Le(1.0f, 1.0f, 1.0f);
	vec3 wLightPos(8.0f, 15.0f, 0.0f);
	light.Create(La, Le, wLightPos);

	///SCENE CREATION
	scene.Create(camera, light, rs);

	
	scene.AddObject(&body);
	scene.AddObject(&head);
	
}

void onExit() {
	glDeleteProgram(shaderProg);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	//draw
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		//lineStrip.AddPoint(cX, cY);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	//camera.Animate(sec);					// animate the camera
	//triangle.Animate(sec);					// animate the triangle object
	head.Animate(sec);
	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

