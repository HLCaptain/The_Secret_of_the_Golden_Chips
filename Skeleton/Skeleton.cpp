//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Püspök-Kiss Balázs
// Neptun : BL6ADS
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

/**
 * This homework is based on the Computer Graphics Sample Program: GPU ray casting.
 * I modified it to fit the description of the second homework.
 */

//#define DEBUG

#include "framework.h"

#ifdef DEBUG
#include <fstream> // needed for real time glsl fragment shader loading
#include <sstream> // needed for real time glsl fragment shader loading
#endif // DEBUG


// vertex shader in GLSL
const char* vertexSource = R"(
	#version 450
	precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye

	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
// fragment shader in GLSL
const char* fragmentSource = R"(
#version 300 es
precision highp float;

struct Material {
	vec3 ka, kd, ks; // ambient, diffuse, specular
	float shininess;
	vec3 F0;
	int rough, reflective;
};

struct Light {
	vec3 position;
	vec3 Le, La;
};

struct Sphere {
	vec3 center;
	float radius;
};

struct Hit {
	float t;
	vec3 position, normal;
	int mat; // material index: 1 = rough, 2 = reflective, <0 = no mat
};

struct Ray {
	vec3 start, dir;
};

struct Dodecahedron {
	vec3 vertices[20];
	int faces[12 * 5];
};

struct Quadratic {
	float a, b, c;
};

// dodecahedron properties
const int numDodFaces = 12 * 5; // 12 faces, 5 vertices for every face
const int numDodVertices = 20;
const float epsilon = 0.0001f;
const float scale = 1.0f;
const float teleportHoleScale = 0.1f; // value between  0..0.45
const vec3 one = vec3(1.0, 1.0, 1.0);
const float PI = 3.1415f;

uniform vec3 wEye;
uniform Light light;
//uniform Material dodDifMat; // mat = 0
//uniform Material dodRefMat; // mat = 1
//uniform Material chipsMat; // mat = 2
uniform Dodecahedron dod;
uniform Quadratic quad;
uniform Material materials[3]; // 0: dodDifMat, 1: dodRefMat, 2: chipsMat
// needed to contain Quadratic
uniform Sphere quadSphere;
uniform float timeMs;

in  vec3 p;					// point on camera window corresponding to the pixel
out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

bool inSphere(vec3 p, Sphere s) {
	return sqrt((p.x - s.center.x) * (p.x - s.center.x) + (p.y - s.center.y) * (p.y - s.center.y) + (p.z - s.center.z) * (p.z - s.center.z)) < s.radius;
}

// from mirascope
Hit solveQuadratic(float a, float b, float c, Ray ray, Hit hit) {
	float discr = b * b - 4.0 * a * c;
	if (discr >= 0.0) {
		float sqrt_discr = sqrt(discr); // calc discriminant
		float t1 = (-b + sqrt_discr) / 2.0 / a;
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		vec3 p1 = ray.start + ray.dir * t1;
		vec3 p2 = ray.start + ray.dir * t2; // two intersections are calculated
		if (!inSphere(p1, quadSphere)) t1 = -1.0; // filtering bad intersections
		if (!inSphere(p2, quadSphere)) t2 = -1.0;
		if (t2 > 0.0 && (t2 < t1 || t1 < 0.0)) t1 = t2; // assigning intersections
		if (t1 > 0.0 && (t1 < hit.t || hit.t < 0.0)) {
			hit.t = t1;
			hit.position = ray.start + ray.dir * hit.t;
			// TODO: negate negated hit.pos.x and y
			hit.normal = normalize(vec3(-2.0 * quad.a * hit.position.x / quad.c, -2.0 * quad.b * hit.position.y / quad.c, 1.0));
			hit.mat = 2; // material is reflective (chips) if hit is valid
		}
	}
	return hit;
}

void getObjPlane(int i, float scale, out vec3 p, out vec3 normal) {
	vec3 p1 = dod.vertices[dod.faces[5 * i + 2] - 1]; // faces[0..19]
	vec3 p2 = dod.vertices[dod.faces[5 * i + 3] - 1]; // vertices[0..59]
	vec3 p3 = dod.vertices[dod.faces[5 * i + 4] - 1];
	normal = cross(p2 - p1, p3 - p1);
	if (dot(p1, normal) < 0.0) normal = -normal;
	p = p1 * scale + vec3(0.0001f, 0.0001f, 0.0001f);
}

vec4 quarternionMul(vec4 q1, vec4 q2) {
	vec4 q;
	q.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
	q.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
	q.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
	q.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
	return q;
}

vec3 rotPointAroundAxis(vec3 point, vec3 axis, float angle) {
	// TODO: SOMETHING
	float s = cos(angle / 2.0);
	float i = axis.x * sin(angle / 2.0);
	float j = axis.y * sin(angle / 2.0);
	float k = axis.z * sin(angle / 2.0);
	vec4 q = vec4(i, j, k, s); // quaternion
	float st = cos(angle / 2.0);
	float it = -axis.x * sin(angle / 2.0);
	float jt = -axis.y * sin(angle / 2.0);
	float kt = -axis.z * sin(angle / 2.0);
	vec4 qt = vec4(it, jt, kt, st); // transposed other quaternion

	float x = point.x;
	float y = point.y;
	float z = point.z;
	vec4 p = vec4(x, y, z, 0);

	vec4 qMulp = quarternionMul(q, p);
	vec4 qpMulqt = quarternionMul(qMulp, qt);
	point = vec3(qpMulqt.x, qpMulqt.y, qpMulqt.z);
	return point;
}

Hit intersectDod(const Ray ray, Hit hit) {
	for	(int i = 0; i < numDodFaces / 5; i++) {
		vec3 p1, normal;
		getObjPlane(i, scale, p1, normal);
		// is normal pointing the same direction as the ray? if no, t = -1
		float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1.0;
		// if we dont hit the plane, then continue
		if (ti <= epsilon || (ti > hit.t && hit.t > 0.0)) continue;
		vec3 pintersect = ray.start + ray.dir * ti;
		// if we hit a face
		bool outsidePortal = false;
		bool outside = false;
		for (int j = 0; j < numDodFaces / 5; j++) {
			if (i == j) continue;
			vec3 p11, n;
			getObjPlane(j, scale - scale * teleportHoleScale, p11, n);
			if (dot(n, pintersect - p11) > 0.0) {
				outsidePortal = true;
			}
			getObjPlane(j, scale, p11, n);
			if (dot(n, pintersect - p11) > 0.0) {
				outside = true;
				break;
			}
		}
		if (!outside) {
			hit.t = ti;
			hit.position = pintersect;
			hit.normal = normalize(normal);
			if (!outsidePortal) hit.mat = 1; else hit.mat = 0; // 0 or 1
			// TODO: check distance from edges and assign mat based on that
		}
	}
	return hit;
}

// from mirascope
Hit intersectQuad(const Ray ray, Hit hit) {
	float A = quad.a * ray.dir.x * ray.dir.x + quad.b  * ray.dir.y * ray.dir.y;
	float B = quad.a * 2.0 * ray.dir.x * ray.start.x + quad.b * 2.0 * ray.dir.y * ray.start.y - quad.c * ray.dir.z;
	float C = quad.a * ray.start.x * ray.start.x + quad.b * ray.start.y * ray.start.y - quad.c * ray.start.z;
	hit = solveQuadratic(A, B, C, ray, hit);
	return hit;
}


Hit firstIntersect(Ray ray) {
	Hit bestHit;
	bestHit.t = -1.0;

	bestHit = intersectQuad(ray, bestHit);
	bestHit = intersectDod(ray, bestHit);

	if (dot(ray.dir, bestHit.normal) > 0.0) bestHit.normal = -bestHit.normal;
	return bestHit;
}

bool shadowIntersect(Ray ray) {	// for directional lights
	//for (int o = 0; o < nObjects; o++) if (intersect(objects[o], ray).t > 0) return true; // hit.t < 0 if no intersection
	Hit bestHit;
	bestHit.t = -1.0;

	bestHit = intersectDod(ray, bestHit);
	bestHit = intersectQuad(ray, bestHit);

	if (bestHit.t > 0.0) return true;
	return false;
}

vec3 Fresnel(vec3 F0, float cosTheta) {
	return F0 + (one - F0) * pow(cosTheta, 5.0);
}

const int maxdepth = 5;

vec3 trace(Ray ray) {
	vec3 weight = vec3(1, 1, 1);
	vec3 outRadiance = vec3(0, 0, 0);
	for (int d = 0; d < maxdepth; d++) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0.0) return weight * light.La;
		if (materials[hit.mat].rough == 1) {
			vec3 lightdir = normalize(light.position - hit.position);
			float cosTheta = dot(hit.normal, lightdir);
			if (cosTheta > 0.0) {
				vec3 LeIn = light.Le / dot(light.position - hit.position, light.position - hit.position);
				outRadiance += weight * LeIn * materials[hit.mat].kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + lightdir);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0.0) {
					outRadiance += weight * LeIn * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
			}
			weight *= materials[hit.mat].ka;
			break;
		}
		if (materials[hit.mat].reflective == 1) {
			weight *= materials[hit.mat].F0 + (one - materials[hit.mat].F0) * pow(1.0 - dot(-ray.dir, hit.normal), 5.0);
			ray.start = hit.position + hit.normal * epsilon;
			vec3 reflectedRay = reflect(ray.dir, hit.normal);
			if (hit.mat == 2) { // we hit the chips
				ray.dir = reflectedRay;
			}
			if (hit.mat == 1) { // we hit the portal
				ray.dir = rotPointAroundAxis(reflectedRay, hit.normal, 72.0 * PI / 180.0);
				ray.start = rotPointAroundAxis(ray.start, hit.normal, 72.0 * PI / 180.0);
			}
		}
	}
	outRadiance += weight * light.La;
	return outRadiance;
}

void main() {
	Ray ray;
	ray.start = wEye; 
	ray.dir = normalize(p - wEye);
	fragmentColor = vec4(trace(ray), 1);
	//if (dot(p, p) < 0.0001) { // middle point for debugging
	//	fragmentColor = vec4(1, 1, 1, 1);
	//}
	//fragmentColor = vec4(0.2, 0.2, 0.2, 0.2);
}
)";

enum MaterialType { ROUGH, REFLECTIVE };

//---------------------------
struct Material {
//---------------------------
	vec3 ka, kd, ks; // ambient, diffuse, specular
	float shininess;
	vec3 F0;
	int rough, reflective;
	MaterialType type;
	Material(const MaterialType& type) : type(type) { }
};

//---------------------------
struct RoughMaterial : Material {
//---------------------------
	RoughMaterial(vec3 _kd = vec3(0.0f, 0.0f, 0.0f), vec3 _ks = vec3(0.0f, 0.0f, 0.0f), float _shininess = 10) : Material(ROUGH) {
		ka = _kd * M_PI; // ambient
		kd = _kd; // diffuse
		ks = _ks; // specular
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

const vec3 one(1, 1, 1);
vec3 operator/(const vec3 num, const vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(const vec3& n = vec3(1.0f, 1.0f, 1.0f), const vec3& kappa = vec3(1.0f, 1.0f, 1.0f)) : Material(REFLECTIVE) {
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
		rough = false;
		reflective = true;
	}
};

//---------------------------
struct Sphere {
//---------------------------
	vec3 center;
	float radius;
	Sphere(const vec3& _center = vec3(0.0f, 0.0f, 0.0f), float _radius = 1.0f) :
		center(_center),
		radius(_radius) { }
};

//---------------------------
struct Quadratic {
//---------------------------
	vec3 center;
	float a, b, c;
	Quadratic(const vec3& _center = vec3(0.0f, 0.0f, 0.0f), const float& _a = 1.0f, const float& _b = 1.0f, const float& _c = 1.0f) :
		center(_center),
		a(_a),
		b(_b),
		c(_c) { }
};

//---------------------------
struct Dodecahedron {
	//---------------------------
	std::vector<vec3> vertices;
	std::vector<int> faces;
	Dodecahedron(const std::vector<vec3>& vertices = std::vector<vec3>(), const std::vector<int>& planes = std::vector<int>()) :
		vertices(vertices),
		faces(planes) { }
};

long curTime = 0;

//---------------------------
struct Camera {
//---------------------------
	vec3 eye, lookat, right, up; // lookat would have been nice to name it "target" smh
	float fov;
	float speed;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tanf(fov / 2);
		up = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt * speed) + (eye.z - lookat.z) * sin(dt * speed) + lookat.x,
			sin(curTime / 4000.0f) / 1.2f,
			-(eye.x - lookat.x) * sin(dt * speed) + (eye.z - lookat.z) * cos(dt * speed) + lookat.z); // rotating around target
		// //eye = vec3((eye.x - lookat.x) * cos(dt * speed) + (eye.z - lookat.z) * sin(dt * speed) + lookat.x,
		//	eye.y /*sin(curTime / 200.0f) / 20 + 45 * M_PI / 180*/,
		//	-(eye.x - lookat.x) * sin(dt * speed) + (eye.z - lookat.z) * cos(dt * speed) + lookat.z); // rotating around target
		//fov = sin(curTime / 200.0f) / 20 + 45 * M_PI / 180;
		set(eye, lookat, up, fov);
	}
	Camera() : speed(1.0f) { }
};

float Fresnel(const float& n, const float& kappa) {
	return ((n - 1) * (n - 1) + kappa * kappa) / ((n + 1) * (n + 1) + kappa * kappa);
}

enum LightType { POINTLIGHT, DIRECTIONAL, LIGHT };

//---------------------------
struct Light {
//---------------------------
	vec3 Le, La; // light energy, light ambient
	LightType type;
	vec3 position;
	virtual vec3 getDir() = 0;
	virtual vec3 getPos() { return position; }
protected:
	Light(LightType _type = LIGHT, vec3 _Le = vec3(0.0f, 0.0f, 0.0f), vec3 _La = vec3(0.0f, 0.0f, 0.0f)) :
		Le(_Le),
		La(_La),
		position(vec3(0, 0, 0)),
		type(_type) { }
};

struct PointLight : public Light {
	PointLight(vec3 _position = vec3(0.0f, 0.0f, 0.0f), vec3 _Le = vec3(0.0f, 0.0f, 0.0f), vec3 _La = vec3(0.0f, 0.0f, 0.0f)) :
		Light(POINTLIGHT, _Le, _La) {
		position = _position;
	}
	vec3 getDir() { return -1 * position; }
	vec3 getPos() { return position; }
};

struct DirectionalLight : public Light {
	vec3 direction;
	DirectionalLight(vec3 _direction = vec3(0.0f, 0.0f, 0.0f), vec3 _Le = vec3(0.0f, 0.0f, 0.0f), vec3 _La = vec3(0.0f, 0.0f, 0.0f)) :
		Light(DIRECTIONAL, _Le, _La),
		direction(normalize(_direction)) { }
	vec3 getDir() { return direction; }
};


// modified from https://www.codespeedy.com/hsv-to-rgb-in-cpp/
vec3 HslToRgb(const float& hue, const float& saturation, const float& lightness) {
	float s = saturation / 100;
	float l = lightness / 100;
	float c = (1 - abs(2 * l - 1) * s);
	float x = c * (1 - abs(fmod(hue / 60.0f, 2) - 1));
	float m = l - c / 2;
	float r, g, b;
	if (hue >= 0 && hue < 60) {
		r = c, g = x, b = 0;
	} else if (hue >= 60 && hue < 120) {
		r = x, g = c, b = 0;
	} else if (hue >= 120 && hue < 180) {
		r = 0, g = c, b = x;
	} else if (hue >= 180 && hue < 240) {
		r = 0, g = x, b = c;
	} else if (hue >= 240 && hue < 300) {
		r = x, g = 0, b = c;
	} else {
		r = c, g = 0, b = x;
	}
	return vec3(r, g, b);
}

//---------------------------
class Shader : public GPUProgram {
//---------------------------
public:
	void setUniformMaterials(const Material& dodDifMat, const Material& dodRefMat, const Material& chipsMat) {
		char name[256];
		// diffuse walls of dod
		sprintf(name, "materials[%d].ka", 0);  setUniform(dodDifMat.ka, name);
		sprintf(name, "materials[%d].kd", 0);  setUniform(dodDifMat.kd, name);
		sprintf(name, "materials[%d].ks", 0);  setUniform(dodDifMat.ks, name);
		sprintf(name, "materials[%d].shininess", 0);  setUniform(dodDifMat.shininess, name);
		sprintf(name, "materials[%d].F0", 0);  setUniform(dodDifMat.F0, name);
		sprintf(name, "materials[%d].rough", 0);  setUniform(dodDifMat.rough, name);
		sprintf(name, "materials[%d].reflective", 0);  setUniform(dodDifMat.reflective, name);
		// reflective portals of dod
		sprintf(name, "materials[%d].ka", 1);  setUniform(dodRefMat.ka, name);
		sprintf(name, "materials[%d].kd", 1);  setUniform(dodRefMat.kd, name);
		sprintf(name, "materials[%d].ks", 1);  setUniform(dodRefMat.ks, name);
		sprintf(name, "materials[%d].shininess", 1);  setUniform(dodRefMat.shininess, name);
		sprintf(name, "materials[%d].F0", 1);  setUniform(dodRefMat.F0, name);
		sprintf(name, "materials[%d].rough", 1);  setUniform(dodRefMat.rough, name);
		sprintf(name, "materials[%d].reflective", 1);  setUniform(dodRefMat.reflective, name);
		// reflective material of quad chips
		sprintf(name, "materials[%d].ka", 2);  setUniform(chipsMat.ka, name);
		sprintf(name, "materials[%d].kd", 2);  setUniform(chipsMat.kd, name);
		sprintf(name, "materials[%d].ks", 2);  setUniform(chipsMat.ks, name);
		sprintf(name, "materials[%d].shininess", 2);  setUniform(chipsMat.shininess, name);
		sprintf(name, "materials[%d].F0", 2);  setUniform(chipsMat.F0, name);
		sprintf(name, "materials[%d].rough", 2);  setUniform(chipsMat.rough, name);
		sprintf(name, "materials[%d].reflective", 2);  setUniform(chipsMat.reflective, name);
	}

	void setUniformLight(const PointLight& light) {
		setUniform(light.La, "light.La");
		setUniform(light.Le, "light.Le");
		setUniform(light.position, "light.position");
	}

	void setUniformCamera(const Camera& camera) {
		setUniform(camera.eye, "wEye");
		setUniform(camera.lookat, "wLookAt");
		setUniform(camera.right, "wRight");
		setUniform(camera.up, "wUp");
	}

	void setUniformObjects(const Quadratic& quad, const Dodecahedron& dod) {
		char name[256];
		for (unsigned int i = 0; i < dod.faces.size(); i++) {
			sprintf(name, "dod.faces[%d]", i);  setUniform(dod.faces[i], name);
		}
		for (unsigned int i = 0; i < dod.vertices.size(); i++) {
			sprintf(name, "dod.vertices[%d]", i);  setUniform(dod.vertices[i], name);
		}

		// maybe useless copying abc
		setUniform(quad.a, "quad.a");
		setUniform(quad.b, "quad.b");
		setUniform(quad.c, "quad.c");

		// uploading the encasing of the quad
		setUniform(vec3(0.0f, 0.0f, 0.0f), "quadSphere.center");
		setUniform(0.3f, "quadSphere.radius");
	}

	void setUniformTime(const long& time) {
		setUniform((float)time, "timeMs");
	}
};

/// Random float number between 0...1
float rnd() { return (float)rand() / RAND_MAX; }

Shader shader; // vertex and fragment shaders

//---------------------------
class Scene {
//---------------------------
	Camera camera;
	RoughMaterial dodDifMat;
	ReflectiveMaterial dodRefMat;
	ReflectiveMaterial chipsMat;
	Dodecahedron dod;
	Quadratic quad;
	PointLight pointLight;
public:
	void build() {

		// INI DODECAHEDRON
		const float g = 0.618f;
		const float G = 1.618f;
		std::vector<vec3> dodecahedronVertices = std::vector<vec3>({
			vec3(0, g, G),
			vec3(0, -g, G),
			vec3(0, -g, -G),
			vec3(0, g, -G),
			vec3(G, 0, g),
			vec3(-G, 0, g),
			vec3(-G, 0, -g),
			vec3(G, 0, -g),
			vec3(g, G, 0),
			vec3(-g, G, 0),
			vec3(-g, -G, 0),
			vec3(g, -G, 0),
			vec3(1, 1, 1),
			vec3(-1, 1, 1),
			vec3(-1, -1, 1),
			vec3(1, -1, 1),
			vec3(1, -1, -1),
			vec3(1, 1, -1),
			vec3(-1, 1, -1),
			vec3(-1, -1, -1),
			});
		std::vector<int> dodecahedronPlanes = std::vector<int>({
			1,2,16,5,13, // f1
			1,13,9,10,14, // f2
			1,14,6,15,2, // f3
			2,15,11,12,16, // f4
			3,4,18,8,17, // f5
			3,17,12,11,20, // f6
			3,20,7,19,4, // f7
			19,10,9,18,4, // f8
			16,12,17,8,5, // f9
			5,8,18,9,13, // f10
			14,10,19,7,6, // f11
			6,7,20,11,15 // f12
			});
		dod.faces = dodecahedronPlanes;
		dod.vertices = dodecahedronVertices;
		// INI DODECAHEDRON END

		// INI CAMERA
		vec3 eye = vec3(0, 0, 1);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 90 * (float)M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		camera.speed = 0.1f;
		// INI CAMERA END

		// INI LIGHT
		pointLight = PointLight(vec3(0.0f, 0.0f, 0.0f), vec3(1.6f, 1.6f, 1.6f), vec3(0.3f, 0.3f, 0.3f));
		// INI LIGHT END

		// INI MATERIALS
		// honey: vec3(1.0f, 0.4f, 0.10f), purple: vec3(6.0f, 0.1f, 0.9f), testgrey: vec3(0.4f, 0.4f, 0.4f)
		// Az arany törésmutatója és kioltási tényezõje: n/k: 0.17/3.1, 0.35/2.7, 1.5/1.9
		vec3 nGold(0.17f, 0.35f, 1.5f), kappaGold(3.1f, 2.7f, 1.9f);
		chipsMat = ReflectiveMaterial(nGold, kappaGold);
		dodDifMat = RoughMaterial(vec3(1.0f, 0.5f, 0.15f), vec3(1.0f, 1.0f, 1.0f), 10.0f); // (kd) diffuse: RGB, (ks) specular: RGB
		dodRefMat = ReflectiveMaterial(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f));
		// INI MATERIALS END

		// INI QUAD
		// quadratic equation: exp(something) - 1 = 0
		// quadratic equation simplified: axx + byy - cz = 0
		quad = Quadratic(vec3(0.2f, 0.2f, 0.2f), 3.2f, 4.2f, 2.8f);
		// INI QUAD END
	}

	void setUniform(Shader& shader) {
		shader.setUniformObjects(quad, dod);
		shader.setUniformMaterials(dodDifMat, dodRefMat, chipsMat);
		shader.setUniformLight(pointLight);
		shader.setUniformCamera(camera);
		//shader.setUniformTime(curTime);
	}

	void Animate(float dt) { 
		camera.Animate(dt);
		float delta = 140 * M_PI / 180;
		vec3 newColor = HslToRgb((cos((float)curTime / 5000.0 + delta) + 1) / 2 * 360, 0, 0);
		dodDifMat.kd = newColor;
		dodDifMat.ks = vec3(1, 1, 1);
		dodDifMat.ka = vec3(1, 1, 1);
	}

	Scene() { }
};

Scene scene;

//---------------------------
class FullScreenTexturedQuad {
//---------------------------
	unsigned int vao = 0;	// vertex array object id and texture id
public:
	void create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

void reloadVerFrag(Shader& sh, const std::string& ver = vertexSource, const std::string& frag = fragmentSource) {
	sh.create(ver.c_str(), frag.c_str(), "fragmentColor");
	sh.Use();
}

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad.create();

#ifdef DEBUG
	// create program for the GPU
	std::ifstream frag("fragmentShader.frag");
	std::stringstream fragSS;
	fragSS << frag.rdbuf();
	reloadVerFrag(shader, vertexSource, fragSS.str());
	frag.close();
#else
	shader.create(vertexSource, fragmentSource, "fragmentColor");
	shader.Use();
#endif // DEBUG
}

// Window has become invalid: Redraw
void onDisplay() {
	static int nFrames = 0;
	nFrames++;
	static long tStart = glutGet(GLUT_ELAPSED_TIME);
	long tEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Frame time: %d msec\r", (tEnd - tStart) / nFrames);

	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	scene.setUniform(shader);
	fullScreenTexturedQuad.Draw();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
#ifdef DEBUG
	if (key == ' ') {
		// reloading fragment and vertex source
		std::ifstream frag("fragmentShader.frag");
		std::stringstream fragSS;
		fragSS << frag.rdbuf();
		reloadVerFrag(shader, vertexSource, fragSS.str());
		frag.close();
	}
#endif // DEBUG
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

long oldTime = 0;
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	curTime = glutGet(GLUT_ELAPSED_TIME);
	if (oldTime == 0) {
		oldTime = curTime;
	}
	long dtime = curTime - oldTime;
	oldTime = curTime;
	scene.Animate((float)dtime / 1000.0f);
	//scene.Animate(0.02f);
	glutPostRedisplay();
}