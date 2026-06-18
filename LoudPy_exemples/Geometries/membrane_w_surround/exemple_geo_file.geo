SetFactory("OpenCASCADE");
Merge "exemple_brep_file.brep";
//+
Physical Curve("submeca_membranne", 18) = {11, 13, 10, 17, 12};
//+
Physical Curve("submeca_surround", 19) = {16, 14, 17, 15};
//+
Physical Curve("subacou_rear", 20) = {1, 2, 3, 4, 5, 16, 12, 11};
//+
Physical Curve("subacou_front", 21) = {13, 9, 14, 6, 8, 7};
//+
Physical Curve("interface_constrained_r_1", 26) = {10};
//+
Physical Curve("interface_constrained_rz_1", 27) = {15};
//+
Physical Curve("interface_forced_z_1", 28) = {12};
//+
Physical Curve("interface_acou_meca_front", 29) = {13, 14};
//+
Physical Curve("interface_acou_meca_rear", 30) = {11, 16, 12};
//+
Physical Curve("PML_r+_1_front", 31) = {7};
//+
Physical Curve("PML_r+_2_rear", 32) = {3};
//+
Physical Curve("PML_z-_2_rear", 33) = {2};
//+
Physical Curve("PML_z+_1_front", 34) = {8};
