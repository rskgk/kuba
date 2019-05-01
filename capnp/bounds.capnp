@0xde961b3161e0684d;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("geom");

using Point = import "point.capnp";

struct Bounds2 {
  min @0 :Point.Point2;
  max @1 :Point.Point2;
}

struct Bounds3 {
  min @0 :Point.Point3;
  max @1 :Point.Point3;
}
