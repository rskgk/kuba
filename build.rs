extern crate capnpc;

// TODO(kgreenek): Don't do this all in the top-level build.rs file.
fn main() {
    ::capnpc::CompilerCommand::new()
        .src_prefix("capnp")
        .file("capnp/bounds.capnp")
        .file("capnp/point.capnp")
        //.file("src/grid_map/grid_map.capnp")
        .run()
        .expect("compiling schema");
}
