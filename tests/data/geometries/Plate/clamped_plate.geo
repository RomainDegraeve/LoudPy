// Clamped circular plate (axisymmetric meridian): radius a = 0.1 m, thickness h = 2 mm.
// Used by Eigen_study/Fourier/Validation_clamped_plate.py (analytical thin-plate modes).
SetFactory("OpenCASCADE");
a = 0.1;
h = 0.002;
Point(1) = {0, 0, 0};
Point(2) = {a, 0, 0};
Point(3) = {a, h, 0};
Point(4) = {0, h, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Physical Curve("submeca_plate", 1) = {1, 2, 3, 4};
Physical Curve("interface_constrained_rz_1", 2) = {2};   // clamped rim
Physical Curve("interface_constrained_r_1", 3) = {4};    // axis (m = 0 condition; m >= 1 automatic)
