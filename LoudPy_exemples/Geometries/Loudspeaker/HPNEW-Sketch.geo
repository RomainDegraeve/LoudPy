SetFactory("OpenCASCADE");
Merge "HPNEW-Sketch.brep";

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
Physical Curve("subacou_front",   9) = {20,10, 11, 12, 13, 5, 4, 3, 2, 1, 113, 22, 21};
Physical Curve("subacou_rear",10) = {
  39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
  54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
  69, 70, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
  91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
  105, 106, 107, 108, 38, 109, 122, 77, 76, 121, 119, 29, 30,
  31, 32, 71, 72, 73, 74, 116, 36, 35, 34, 115, 27, 26, 25,
  24, 23, 9, 18, 110, 17, 16, 15, 33, 14, 7, 8, 111
};

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

// --- PML ---
Physical Curve("PML_r+_1_front", 17) = {12};
Physical Curve("PML_r+_2_rear",  18) = {16};
Physical Curve("PML_z-_2_rear",  19) = {17};
Physical Curve("PML_z+_1_front", 20) = {11};
//+
Physical Curve("interface_acou_meca_rear", 15) += {8, 7, 33};
//+
Physical Curve(" interface_acou_meca_front", 16) -= {123};
//+
Physical Curve(" interface_acou_meca_rear", 15) -= {123};
//+
Physical Curve(" interface_acou_meca_rear", 15) -= {109};
//+

