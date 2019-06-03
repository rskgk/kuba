extern crate kuba;

fn main() {
    println!("READING");
    let point_clouds = kuba::kitti::point_cloud_reader::read_from_dir(
        std::path::Path::new(
            "/Users/kevin/Downloads/2011_09_26/2011_09_26_drive_0087_sync/velodyne_points/data",
        ),
        true,
    )
    .unwrap();
    println!("DONE READING");
    for point in point_clouds[0].points_iter() {
        println!("{}", point);
    }
}
