// Clamped circular plate (axisymmetric meridian) with its dimensions as parameters: radius a, thickness h.
// DefineConstant: a model parameter of the same name (lp.Parameters(a=…, h=…)) replaces these values, so
// lp.Sweep can sweep the geometry. Used by Sweep_study/Running_plate.py.
SetFactory("OpenCASCADE");
DefineConstant[ a = 0.1, h = 0.002 ];
Point(1) = {0, 0, 0};
Point(2) = {a, 0, 0};
Point(3) = {a, h, 0};
Point(4) = {0, h, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Physical Curve("plate") = {1, 2, 3, 4};
Physical Curve("rim") = {2};       // clamped
Physical Curve("axis") = {4};      // symmetry axis (u_r = 0)
