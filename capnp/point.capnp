@0xc2c60d0e4d404bda;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("geom");

struct Point2 {
  x @0 :Float32;
  y @1 :Float32;
}

struct Point3 {
  x @0 :Float32;
  y @1 :Float32;
  z @2 :Float32;
}
