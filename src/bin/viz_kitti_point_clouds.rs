extern crate kuba;

fn main() {
    let kitti_dataset_path =
        std::path::Path::new("/Users/kevin/Downloads/2011_09_26/2011_09_26_drive_0087_sync");
    println!("Reading poses...");
    let _poses = kuba::kitti::oxt_reader::read_from_dir(
        &kitti_dataset_path.join("oxts/data"),
        false,
    )
    .unwrap();
    println!("Reading points clouds...");
    let _point_clouds = kuba::kitti::point_cloud_reader::read_from_dir(
        &kitti_dataset_path.join("velodyne_points/data"),
        false,
    )
    .unwrap();
    println!("Done!");
}
