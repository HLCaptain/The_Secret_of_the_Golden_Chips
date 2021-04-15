#version 330
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
	if (discr >= 0) {
		float sqrt_discr = sqrt(discr); // calc discriminant
		float t1 = (-b + sqrt_discr) / 2.0 / a;
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		vec3 p1 = ray.start + ray.dir * t1;
		vec3 p2 = ray.start + ray.dir * t2; // two intersections are calculated
		if (!inSphere(p1, quadSphere)) t1 = -1; // filtering bad intersections
		if (!inSphere(p2, quadSphere)) t2 = -1;
		if (t2 > 0 && (t2 < t1 || t1 < 0)) t1 = t2; // assigning intersections
		if (t1 > 0 && (t1 < hit.t || hit.t < 0)) {
			hit.t = t1;
			hit.position = ray.start + ray.dir * hit.t;
			// TODO: negate negated hit.pos.x and y
			hit.normal = normalize(vec3(-2 * quad.a * hit.position.x / quad.c, -2 * quad.b * hit.position.y / quad.c, 1));
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
	if (dot(p1, normal) < 0) normal = -normal;
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
	float s = cos(angle / 2);
	float i = axis.x * sin(angle / 2);
	float j = axis.y * sin(angle / 2);
	float k = axis.z * sin(angle / 2);
	vec4 q = vec4(i, j, k, s); // quaternion
	float st = cos(angle / 2);
	float it = -axis.x * sin(angle / 2);
	float jt = -axis.y * sin(angle / 2);
	float kt = -axis.z * sin(angle / 2);
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
		float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
		// if we dont hit the plane, then continue
		if (ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;
		vec3 pintersect = ray.start + ray.dir * ti;
		// if we hit a face
		bool outsidePortal = false;
		bool outside = false;
		for (int j = 0; j < numDodFaces / 5; j++) {
			if (i == j) continue;
			vec3 p11, n;
			getObjPlane(j, scale - scale * teleportHoleScale, p11, n);
			if (dot(n, pintersect - p11) > 0) {
				outsidePortal = true;
			}
			getObjPlane(j, scale, p11, n);
			if (dot(n, pintersect - p11) > 0) {
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
	float B = quad.a * 2 * ray.dir.x * ray.start.x + quad.b * 2 * ray.dir.y * ray.start.y - quad.c * ray.dir.z;
	float C = quad.a * ray.start.x * ray.start.x + quad.b * ray.start.y * ray.start.y - quad.c * ray.start.z;
	hit = solveQuadratic(A, B, C, ray, hit);
	return hit;
}


Hit firstIntersect(Ray ray) {
	Hit bestHit;
	bestHit.t = -1;

	bestHit = intersectQuad(ray, bestHit);
	bestHit = intersectDod(ray, bestHit);

	if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = -bestHit.normal;
	return bestHit;
}

bool shadowIntersect(Ray ray) {	// for directional lights
	//for (int o = 0; o < nObjects; o++) if (intersect(objects[o], ray).t > 0) return true; // hit.t < 0 if no intersection
	Hit bestHit;
	bestHit.t = -1;

	bestHit = intersectDod(ray, bestHit);
	bestHit = intersectQuad(ray, bestHit);

	if (bestHit.t > 0) return true;
	return false;
}

vec3 Fresnel(vec3 F0, float cosTheta) {
	return F0 + (one - F0) * pow(cosTheta, 5);
}

const int maxdepth = 5;

vec3 trace(Ray ray) {
	vec3 weight = vec3(1, 1, 1);
	vec3 outRadiance = vec3(0, 0, 0);
	for (int d = 0; d < maxdepth; d++) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return weight * light.La;
		if (materials[hit.mat].rough == 1) {
			vec3 lightdir = normalize(light.position - hit.position);
			float cosTheta = dot(hit.normal, lightdir);
			if (cosTheta > 0) {
				vec3 LeIn = light.Le / dot(light.position - hit.position, light.position - hit.position);
				outRadiance += weight * LeIn * materials[hit.mat].kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + lightdir);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) {
					outRadiance += weight * LeIn * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
			}
			weight *= materials[hit.mat].ka;
			break;
		}
		if (materials[hit.mat].reflective == 1) {
			weight *= materials[hit.mat].F0 + (one - materials[hit.mat].F0) * pow(1 - dot(-ray.dir, hit.normal), 5);
			ray.start = hit.position + hit.normal * epsilon;
			vec3 reflectedRay = reflect(ray.dir, hit.normal);
			if (hit.mat == 2) { // we hit the chips
				ray.dir = reflectedRay;
			}
			if (hit.mat == 1) { // we hit the portal
				ray.dir = rotPointAroundAxis(reflectedRay, hit.normal, 72 * PI / 180 + 0*timeMs / 100 * PI / 180);
				ray.start = rotPointAroundAxis(ray.start, hit.normal, 72 * PI / 180 + 0*timeMs / 100 * PI / 180);
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
	if (dot(p, p) < 0.0001) {
		fragmentColor = vec4(1, 1, 1, 1);
	}
	//fragmentColor = vec4(0.2, 0.2, 0.2, 0.2);
}