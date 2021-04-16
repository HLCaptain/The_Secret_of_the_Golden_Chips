#version 300 es
precision highp float;

// from SzKL
struct Material {
	vec3 ka, kd, ks; // ambient, diffuse, specular
	float shininess;
	vec3 F0;
	int rough, reflective;
};

// from SzKL
struct Light {
	vec3 position;
	vec3 Le, La;
};

// from SzKL
struct Sphere {
	vec3 center;
	float radius;
};

// from SzKL
struct Hit {
	float t;
	vec3 position, normal;
	int mat; // material index: 0 = dod dif, 1 = dod ref, 2 = chips ref
};

// from SzKL
struct Ray {
	vec3 start, dir;
};

// containing the 3 elements of a quadratic equation
struct Quadratic {
	float a, b, c;
};

// dodecahedron properties
const int numDodFaces = 12 * 5; // 12 faces, 5 vertices for every face
const int numDodVertices = 20; // number of vertices a dodecahedron has
const float epsilon = 0.0001f; // small offset
const float scale = 1.0f; // scale of the dodecahedron
const float teleportHoleScale = 0.100f; // distance from edges (in units (=meters))
const vec3 one = vec3(1.0, 1.0, 1.0); // constant one vector
const float PI = 3.1415f; // bad approximation of PI
const int maxdepth = 5; // max 5 bounces

uniform vec3 wEye;
uniform Light light;
uniform vec3 dodVertices[20];
uniform	int dodFaces[12 * 5];
uniform Quadratic quad;
uniform Material materials[3]; // 0: dodDifMat, 1: dodRefMat, 2: chipsMat
// needed to contain Quadratic
uniform Sphere quadSphere;
uniform float timeMs; // current time in ms, can do some dope things if used

in  vec3 p;					// point on camera window corresponding to the pixel
out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

// checks if p point is in s sphere
bool inSphere(vec3 p, Sphere s) {
	return sqrt((p.x - s.center.x) * (p.x - s.center.x) + (p.y - s.center.y) * (p.y - s.center.y) + (p.z - s.center.z) * (p.z - s.center.z)) < s.radius;
}

// from mirascope
Hit solveQuadratic(float a, float b, float c, Ray ray, Hit hit) {
	float discr = b * b - 4.0 * a * c; // calc discriminant
	if (discr >= 0.0) {
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		vec3 p1 = ray.start + ray.dir * t1; // the two intersections are calculated
		vec3 p2 = ray.start + ray.dir * t2; // the two intersections are calculated
		if (!inSphere(p1, quadSphere)) t1 = -1.0; // filtering bad intersections
		if (!inSphere(p2, quadSphere)) t2 = -1.0;
		if (t2 > 0.0 && (t2 < t1 || t1 < 0.0)) t1 = t2; // t1 is now the closest to the camera
		if (t1 > 0.0 && (t1 < hit.t || hit.t < 0.0)) { // check if t1 is valid
			hit.t = t1;
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = normalize(vec3(-2.0 * quad.a * hit.position.x / quad.c, -2.0 * quad.b * hit.position.y / quad.c, 1.0));
			hit.mat = 2; // material is reflective (chips) if hit is valid
		}
	}
	return hit;
}

// modified from mirascope
void getObjPlane(int i, float scale, out vec3 p, out vec3 normal) {
	vec3 p1 = dodVertices[dodFaces[5 * i] - 1]; // faces[0..19]
	vec3 p2 = dodVertices[dodFaces[5 * i + 1] - 1]; // vertices[0..59]
	vec3 p3 = dodVertices[dodFaces[5 * i + 2] - 1];
	normal = cross(p2 - p1, p3 - p1); // linearly independent vector's cross on the surface equals the surface's normal vector
	if (dot(p1, normal) < 0.0) normal = -normal;
	p = p1 * scale + vec3(0.0000f, 0.0000f, 0.0000f);
}

// calculating the distance between two 3D points
float distanceVec3(vec3 p1, vec3 p2) {
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

// modified code from https://math.stackexchange.com/questions/1905533/find-perpendicular-distance-from-point-to-line-in-3d
float distanceFromLine(vec3 a, vec3 b, vec3 c) {
	// a: point; b,c: defines the line
	vec3 d = normalize(c - b);
	vec3 v = a - b;
	float t = dot(v, d);
	vec3 p = b + t * d;
	return distanceVec3(p, a);
}

// checks if p is nearby the edges of that plane
bool isInDodPlane(int n, float dist, vec3 p) {
	// checking last and first point
	vec3 p1 = dodVertices[dodFaces[5 * n + 4] - 1];
	vec3 p2 = dodVertices[dodFaces[5 * n] - 1];
	if (distanceFromLine(p, p1, p2) < dist) { // check if the point is nearby the edge
		return true;
	}
	// checking other points
	for (int i = 1; i < 5; i++) {
		p1 = dodVertices[dodFaces[5 * n + i - 1] - 1];
		p2 = dodVertices[dodFaces[5 * n + i] - 1];
		if (distanceFromLine(p, p1, p2) < dist) { // check if the point is nearby the edge
			return true;
		}
	}
	return false;
}

// quarternion multiplication
vec4 quarternionMul(vec4 q1, vec4 q2) {
	// got some help from a Discord server with this from "Bálint"
	vec4 q;
	q.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
	q.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
	q.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
	q.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
	return q;
}

vec3 rotPointAroundAxis(vec3 point, vec3 axis, float angle) {
	// setting up the two needed quarternion
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
	// setting up the point to rotate
	float x = point.x;
	float y = point.y;
	float z = point.z;
	vec4 p = vec4(x, y, z, 0);
	// q * p
	vec4 qMulp = quarternionMul(q, p);
	// (q * p) * qt
	vec4 qpMulqt = quarternionMul(qMulp, qt);
	// getting the rotated point
	point = vec3(qpMulqt.x, qpMulqt.y, qpMulqt.z);
	return point;
}

Hit intersectDod(const Ray ray, Hit hit) {
	for	(int i = 0; i < numDodFaces / 5; i++) { // numDodFaces is 60, so /5 is fine
		vec3 p1, normal;
		getObjPlane(i, scale, p1, normal);
		// is normal pointing the same direction as the ray? if no, t = -1
		float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1.0;
		// if we dont hit the plane, then continue
		if (ti <= epsilon || (ti > hit.t && hit.t > 0.0)) continue;
		vec3 pintersect = ray.start + ray.dir * ti; // if we hit it, then calculate the intersected point's position
		// if we hit a face
		bool outside = false;
		for (int j = 0; j < numDodFaces / 5; j++) { // numDodFaces is 60, so /5 is fine
			if (i == j) continue;
			vec3 p11, n;
			getObjPlane(j, scale, p11, n);
			if (dot(n, pintersect - p11) > 0.0) {
				outside = true;
				break;
			}
		}
		if (!outside) { // if the point is not outside of the plane
			hit.t = ti;
			hit.position = pintersect;
			hit.normal = normalize(normal);
			bool insidePortal = isInDodPlane(i, teleportHoleScale, hit.position);
			// check distance from edges and assign mat based on that
			if (insidePortal) hit.mat = 0; else hit.mat = 1; // 0 or 1	
		}
	}
	return hit;
}

// from mirascope simulator (heavily modified)
Hit intersectQuad(const Ray ray, Hit hit) {
	float A = quad.a * ray.dir.x * ray.dir.x + quad.b  * ray.dir.y * ray.dir.y;
	float B = quad.a * 2.0 * ray.dir.x * ray.start.x + quad.b * 2.0 * ray.dir.y * ray.start.y - quad.c * ray.dir.z;
	float C = quad.a * ray.start.x * ray.start.x + quad.b * ray.start.y * ray.start.y - quad.c * ray.start.z;
	hit = solveQuadratic(A, B, C, ray, hit);
	return hit;
}

// calculating obj intersetion
Hit firstIntersect(Ray ray) {
	Hit bestHit;
	bestHit.t = -1.0; // always initiate hit as invalid

	// methods only overwrite bestHit if the hit would be closer to the camera
	bestHit = intersectQuad(ray, bestHit);
	bestHit = intersectDod(ray, bestHit);

	if (dot(ray.dir, bestHit.normal) > 0.0) bestHit.normal = -bestHit.normal;
	return bestHit;
}

// calculating shadow
bool shadowIntersect(Ray ray) {
	Hit bestHit;
	bestHit.t = -1.0; // always initiate hit as invalid

	// methods only overwrite bestHit if the hit would be closer to the camera
	bestHit = intersectDod(ray, bestHit);
	bestHit = intersectQuad(ray, bestHit);

	if (bestHit.t > 0.0) return true;
	return false;
}

vec3 Fresnel(vec3 F0, float cosTheta) {
	return F0 + (one - F0) * pow(cosTheta, 5.0);
}

// modified trace from mirascope
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
				ray.dir = rotPointAroundAxis(reflectedRay, hit.normal, 72.0 * PI / 180.0 + 0.0* timeMs * PI / 180.0 / 200.0);
				ray.start = rotPointAroundAxis(ray.start, hit.normal, 72.0 * PI / 180.0 + 0.0* timeMs * PI / 180.0 / 200.0);
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
	//if (dot(p, p) < 0.0001) fragmentColor = vec4(1, 1, 1, 1); // middle point for debugging
}