@0xb509c75cd299807a;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("grid_map");

using Bounds = import "geom/bounds.capnp";

Struct GridMap2f {
  resolution @0 :Float64;
  bounds @1 :Bounds.Bounds2;
  data @2 :List(List(Float32));
}

Struct GridMap3f {
  resolution @0 :Float64;
  bounds @1 :Bounds.Bounds3;
  data @2 :List(List(Float32));
}
