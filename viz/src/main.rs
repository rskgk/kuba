extern crate kiss3d;
extern crate kuba;
extern crate nalgebra as na;

use kuba::prelude::*;

fn draw_frame_marker(window: &mut kiss3d::window::Window, pose: &kuba::Pose3, length: f32) {
    let origin = pose.transform_point(&na::Point3::origin());
    window.draw_line(
        &origin,
        &(origin + pose.transform_vector(&na::Vector3::new(length, 0.0, 0.0))),
        &na::Point3::new(1.0, 0.0, 0.0),
    );
    window.draw_line(
        &origin,
        &(origin + pose.transform_vector(&na::Vector3::new(0.0, length, 0.0))),
        &na::Point3::new(0.0, 1.0, 0.0),
    );
    window.draw_line(
        &origin,
        &(origin + pose.transform_vector(&na::Vector3::new(0.0, 0.0, length))),
        &na::Point3::new(0.0, 0.0, 1.0),
    );
}

struct AppState {
    poses: Vec<kuba::Pose3>,
    point_clouds: Vec<kuba::PointCloud3>,
    index: usize,
    loop_counter: usize,
    first_loop: bool,
    grid_map: kuba::LidarOccupancyGridMap3,
}

impl kiss3d::window::State for AppState {
    fn step(&mut self, window: &mut kiss3d::window::Window) {
        if self.first_loop || self.loop_counter > 1 {
            if !self.first_loop {
                self.loop_counter = 0;
                self.index = (self.index + 1) % self.point_clouds.len();
            }
            self.first_loop = false;
            let origin = kuba::Point3::from(self.poses[self.index].translation.vector);
            let start_t = std::time::Instant::now();
            self.grid_map
                .integrate_point_cloud(&origin, &self.point_clouds[self.index]);
            println!("integrate_point_cloud ms: {}", start_t.elapsed().as_millis());
        }
        let start_t = std::time::Instant::now();
        let shape = self.grid_map.grid_map.data.shape();
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                for z in 0..shape[2] {
                    let cell = kuba::cell3![x as isize, y as isize, z as isize];
                    if self.grid_map.occupied(&cell) {
                        let point = self.grid_map.point_from_cell(&cell);
                        window.draw_point(&point, &na::Point3::new(0.8, 0.0, 0.0));
                    }
                }
            }
        }
        println!("draw point cloud ms: {}", start_t.elapsed().as_millis());
        for point in self.point_clouds[self.index].points_iter() {
            window.draw_point(&point, &na::Point3::new(0.0, 0.6, 0.8));
        }
        draw_frame_marker(window, &self.poses[self.index], 1.0);
        self.loop_counter += 1;
    }
}

fn main() {
    // TODO(kgreenek): Get this from a commandline arg.
    //let kitti_dataset_path =
    //    std::path::Path::new("/home/kevin/data/kitti/2011_09_26/2011_09_26_drive_0002_sync");
    let kitti_dataset_path =
        std::path::Path::new("/Users/kevin/Downloads/2011_09_26/2011_09_26_drive_0087_sync/");
    println!("Reading poses...");
    let poses =
        kuba::kitti::oxt_reader::read_from_dir(&kitti_dataset_path.join("oxts/data"), false)
            .unwrap();
    println!("Reading point clouds...");
    let point_clouds = kuba::kitti::point_cloud_reader::read_from_dir(
        &kitti_dataset_path.join("velodyne_points/data"),
        false,
    )
    .unwrap();
    assert!(point_clouds.len() == poses.len());
    let point_clouds = point_clouds
        .into_iter()
        .zip(&poses)
        .map(|(point_cloud, pose)| point_cloud.transform(&pose.to_homogeneous()))
        .collect();
    let grid_map = kuba::LidarOccupancyGridMap3::default();
    println!("Opening window...");
    let mut window = kiss3d::window::Window::new("Kitti Point Cloud Vizualizer");
    window.set_light(kiss3d::light::Light::StickToCamera);
    window.set_point_size(1.0);
    let state = AppState {
        poses: poses,
        point_clouds: point_clouds,
        index: 0,
        loop_counter: 0,
        first_loop: true,
        grid_map: grid_map,
    };
    window.render_loop(state);
}
