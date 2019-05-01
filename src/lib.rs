extern crate approx;
extern crate nalgebra as na;
extern crate ndarray as nd;

pub mod geom;
pub use geom::*;

pub mod maps;
pub use maps::*;

extern crate capnp;

pub mod point_capnp {
  include!(concat!(env!("OUT_DIR"), "/point_capnp.rs"));
}

pub mod bounds_capnp {
  include!(concat!(env!("OUT_DIR"), "/bounds_capnp.rs"));
}
