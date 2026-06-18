SetFactory("OpenCASCADE");
Merge "exemple_brep_file.brep";
//+
Physical Curve("submeca_membranne", 16) = {7, 8, 9, 6};
//+
Physical Curve("submeca_surround", 17) = {15, 1, 8, 14};
//+
Physical Curve("interface_constrained_rz_", 18) = {14};
//+
Physical Curve("interface_constrained_r_", 19) = {6};
//+
Physical Curve("interface_forced_z_", 20) = {8};
//+
Physical Curve("subacou_rear", 21) = {11, 12, 13, 15, 7, 10};
//+
Physical Curve("subacou_front", 22) = {9, 2, 1, 5, 4, 3};
//+
Physical Curve("interface_acou_meca_front", 23) = {9};
//+
Physical Curve("interface_acou_meca_rear", 24) = {7};
//+
Physical Curve("PML_r+_1_front", 27) = {3};
//+
Physical Curve("PML_r+_2_rear", 28) = {12};
//+
Physical Curve("PML_z-_2_rear", 26) = {11};
//+
Physical Curve("PML_z+_1_front", 25) = {4};

