SetFactory("OpenCASCADE");
Merge "HPNEW-Sketch.brep";

// Radius of the hemispherical exterior boundary (DtN condition), centred on
// the baffle plane z = 0.032 m on the axis.  Must enclose the loudspeaker and
// every receiver where the pressure is read.
R_dtn = 0.2;

// Remove the rectangular outer boundary (axis, top, side, baffle segments)
Recursive Delete { Curve{11}; Curve{12}; }
Delete { Curve{10}; Curve{13}; }

Point(1001) = {0,     0.032 + R_dtn, 0};   // top of the arc, on the axis
Point(1002) = {R_dtn, 0.032,         0};   // foot of the arc, on the baffle
Point(1003) = {0,     0.032,         0};   // arc centre
Line(1001)   = {13, 1001};                 // axis   (0, 0.013) -> (0, 0.032+R)
Circle(1002) = {1001, 1003, 1002};         // hemispherical boundary
Line(1003)   = {6, 1002};                  // rigid baffle (0.0574, 0.032) -> (R, 0.032)
Recursive Delete { Point{1003}; }


// --- SUBMECA ---
Physical Curve("submeca_membranne", 1) = {113, 112, 33, 32, 23, 24, 28, 29, 30, 31, 114};
Physical Curve("submeca_dustcap",   2) = {20, 9, 114, 22, 21, 19};
Physical Curve("submeca_surround",  3) = {8, 3, 2, 1, 112, 6, 7, 4, 5};
Physical Curve("submeca_former",    4) = {26, 25, 28, 119, 118, 117, 116, 37, 27, 115};
Physical Curve("submeca_coil",      5) = {37, 34, 35, 36};
Physical Curve("submeca_glue_1",    6) = {118, 111, 120, 75};
Physical Curve("submeca_spider",    7) = {
  76, 72, 80, 68, 64, 84, 88, 60, 56, 92, 77, 71, 96, 52, 79, 69,
  78, 81, 67, 100, 83, 48, 70, 65, 82, 85, 63, 87, 44, 104, 66,
  61, 86, 108, 40, 89, 59, 91, 57, 62, 90, 55, 93, 95, 58, 53,
  94, 51, 97, 99, 54, 49, 98, 101, 47, 103, 45, 50, 102, 43,
  105, 107, 46, 41, 106, 42, 121, 73, 74, 75, 120, 117, 123,
  39, 122, 38
};
Physical Curve("submeca_glue_2",    8) = {110, 123, 109};

// --- SUBACOU ---
Physical Curve("subacou_front",   9) = {20, 1001, 1002, 1003, 5, 4, 3, 2, 1, 113, 22, 21};

// --- EXTERIOR BOUNDARY: exact non-reflecting (DtN) condition ---
Physical Curve("dtn_boundary", 30) = {1002};

// --- INTERFACES ---
Physical Curve("interface_constrained_r_1", 11) = {19};
Physical Curve("interface_constrained_rz_1",12) = {109};
Physical Curve("interface_constrained_rz_2",13) = {6};
Physical Curve("interface_forced_z_1",      14) = {37};

Physical Curve("interface_acou_meca_rear",  15) = {
  32, 39, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
  51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
  65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 80, 81, 82,
  83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
  97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
  109, 110, 122, 123, 31, 30, 29, 119, 111, 121, 73, 74,
  72, 9, 23, 24, 25, 26, 27, 115, 34, 35, 36, 116
};

Physical Curve("interface_acou_meca_front", 16) = {20, 21, 22, 113,  1,2, 3, 4, 5};


//+
Physical Curve("interface_acou_meca_rear", 15) += {8, 7, 33};
//+
Physical Curve(" interface_acou_meca_front", 16) -= {123};
//+
Physical Curve(" interface_acou_meca_rear", 15) -= {123};
//+
Physical Curve(" interface_acou_meca_rear", 15) -= {109};
//+

//+
Recursive Delete {
  Curve{18}; Curve{16}; Curve{17}; 
}

//+
Recursive Delete {
  Curve{15}; Curve{14}; 
}
