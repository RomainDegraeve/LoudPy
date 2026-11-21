SetFactory("OpenCASCADE");
Merge "HPNEW-Sketch.brep";

// Front AND rear radiation, each half-space closed by the exact DtN condition.
//
// Same 1 mm rigid baffle as HPNEW-Sketch.geo (forced FSI reference):
//   front air above z = 0.032 m (baffle face from r = 0.05735 m),
//   rear  air below z = 0.031 m (baffle face from r = 0.054349 m).
// Each half-space is closed by a hemisphere centred on its own baffle face:
//   front: centre (0, 0.032), radius R_front
//   rear : centre (0, 0.031), radius R_rear
//
// loudpy naming conventions (Problem.orientations_key / directions_key):
//   "front" / "rear" in an acoustic sub-domain or interface name gives the
//   orientation +1 / -1 (rear acoustic matrices, rear fluid-structure coupling
//   and rear DtN terms are multiplied by -1); "_r_", "_z_", "_rz_" give the
//   directions of the clamped / forced interfaces.

R_front = 0.2;
R_rear  = 0.2;

// Remove the rectangular PML boxes of HPNEW-Sketch.brep
//   front: axis 10, top 11, side 12, baffle 13
//   rear : baffle 15, side 16, bottom 17, axis 18   (14 = baffle edge, kept)
Recursive Delete { Curve{11}; Curve{12}; }
Delete { Curve{10}; Curve{13}; }
Recursive Delete { Curve{16}; Curve{17}; }
Delete { Curve{15}; Curve{18}; }

// --- FRONT exterior: hemisphere on the front baffle face (z = 0.032) ---
Point(1001) = {0,       0.032 + R_front, 0};   // top of the arc, on the axis
Point(1002) = {R_front, 0.032,           0};   // foot of the arc, on the baffle
Point(1003) = {0,       0.032,           0};   // arc centre
Line(1001)   = {13, 1001};                     // axis    (0, 0.013) -> (0, 0.032 + R)
Circle(1002) = {1001, 1003, 1002};             // front hemisphere
Line(1003)   = {6, 1002};                      // front baffle (0.05735, 0.032) -> (R, 0.032)
Recursive Delete { Point{1003}; }

// --- REAR exterior: hemisphere on the rear baffle face (z = 0.031) ---
Point(1011) = {R_rear, 0.031,          0};     // foot of the arc, on the baffle
Point(1012) = {0,      0.031 - R_rear, 0};     // bottom of the arc, on the axis
Point(1013) = {0,      0.031,          0};     // arc centre
Line(1011)   = {16, 1011};                     // rear baffle (0.054349, 0.031) -> (R, 0.031)
Circle(1012) = {1011, 1013, 1012};             // rear hemisphere
Line(1013)   = {11, 1012};                     // axis    (0, 0.012) -> (0, 0.031 - R)
Recursive Delete { Point{1013}; }


// --- SUBMECA ---
Physical Curve("cone", 1) = {113, 112, 33, 32, 23, 24, 28, 29, 30, 31, 114};
Physical Curve("dustcap",   2) = {20, 9, 114, 22, 21, 19};
Physical Curve("surround",  3) = {8, 3, 2, 1, 112, 6, 7, 4, 5};
Physical Curve("former",    4) = {26, 25, 28, 119, 118, 117, 116, 37, 27, 115};
Physical Curve("coil",      5) = {37, 34, 35, 36};
Physical Curve("glue_1",    6) = {118, 111, 120, 75};
Physical Curve("spider",    7) = {
  76, 72, 80, 68, 64, 84, 88, 60, 56, 92, 77, 71, 96, 52, 79, 69,
  78, 81, 67, 100, 83, 48, 70, 65, 82, 85, 63, 87, 44, 104, 66,
  61, 86, 108, 40, 89, 59, 91, 57, 62, 90, 55, 93, 95, 58, 53,
  94, 51, 97, 99, 54, 49, 98, 101, 47, 103, 45, 50, 102, 43,
  105, 107, 46, 41, 106, 42, 121, 73, 74, 75, 120, 117, 123,
  39, 122, 38
};
Physical Curve("glue_2",    8) = {110, 123, 109};

// --- SUBACOU (front: orientation +1, rear: orientation -1) ---
Physical Curve("air_front",  9) = {20, 1001, 1002, 1003, 5, 4, 3, 2, 1, 113, 22, 21};
Physical Curve("air_rear",  10) = {
  39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
  54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
  69, 70, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
  91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
  105, 106, 107, 108, 38, 109, 122, 77, 76, 121, 119, 29, 30,
  31, 32, 71, 72, 73, 74, 116, 36, 35, 34, 115, 27, 26, 25,
  24, 23, 9, 1013, 110, 1012, 1011, 33, 14, 7, 8, 111
};

// --- EXTERIOR BOUNDARIES: exact non-reflecting (DtN) condition ---
Physical Curve("dtn_front", 30) = {1002};
Physical Curve("dtn_rear",  31) = {1012};

// --- INTERFACES ---
Physical Curve("axis", 11) = {19};
Physical Curve("spider_rim",12) = {109};
Physical Curve("surround_rim",13) = {6};
Physical Curve("coil_drive",      14) = {37};

Physical Curve("wet_rear",  15) = {
  32, 39, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
  51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
  65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 80, 81, 82,
  83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
  97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
  109, 110, 122, 123, 31, 30, 29, 119, 111, 121, 73, 74,
  72, 9, 23, 24, 25, 26, 27, 115, 34, 35, 36, 116
};

Physical Curve("wet_front", 16) = {20, 21, 22, 113,  1,2, 3, 4, 5};

// Same edits as HPNEW-Sketch.geo (the forced FSI reference)
Physical Curve("wet_rear", 15) += {8, 7, 33};
Physical Curve("wet_front", 16) -= {123};
Physical Curve("wet_rear", 15) -= {123};
Physical Curve("wet_rear", 15) -= {109};
