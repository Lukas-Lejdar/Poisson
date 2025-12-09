r0 = 0.5;
r1 = 0.75;
r2 = 1.0;
theta = Pi/6;
lc = 0.3;

c = {0,0,0};

p0_s = {r0,0,0};
p1_s = {r0*Cos(theta), r0*Sin(theta),0};

p0_m = {r1,0,0};
p1_m = {r1*Cos(theta), r1*Sin(theta),0};

p0_l = {r2,0,0};
p1_l = {r2*Cos(theta), r2*Sin(theta),0};

Point(1) = {p0_s[0],p0_s[1],0, lc};
Point(2) = {p1_s[0],p1_s[1],0, lc};
Point(3) = {p0_m[0],p0_m[1],0, lc};
Point(4) = {p1_m[0],p1_m[1],0, lc};
Point(5) = {p0_l[0],p0_l[1],0, lc};
Point(6) = {p1_l[0],p1_l[1],0, lc};
Point(7) = {0,0,0, lc};

Circle(1) = {1,7,2};
Circle(2) = {3,7,4};
Circle(3) = {5,7,6};

Line(4) = {1,3};
Line(5) = {2,4};
Line(6) = {3,5};
Line(7) = {4,6};

Line Loop(1) = {1,5,-2,-4};
Line Loop(2) = {2,7,-3,-6}; // outer sector

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Recombine Surface {1,2};

Physical Curve("inner_circle") = {1};
//Physical Curve("middle_circle") = {2};
Physical Curve("outer_circle") = {3};
Physical Curve("radial_lines") = {4,5,6,7};

Physical Surface("domain_inner") = {1};
Physical Surface("domain_outer") = {2};


