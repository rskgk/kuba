extern crate kiss3d;
extern crate kuba;
extern crate nalgebra as na;

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
}

impl kiss3d::window::State for AppState {
    fn step(&mut self, window: &mut kiss3d::window::Window) {
        if self.loop_counter >= 10 {
            self.loop_counter = 0;
            self.index = (self.index + 1) % self.point_clouds.len();
        }
        self.loop_counter += 1;
        for point in self.point_clouds[self.index].points_iter() {
            window.draw_point(&point, &na::Point3::new(0.0, 0.6, 0.8));
        }
        draw_frame_marker(window, &self.poses[self.index], 1.0);
    }
}

fn main() {
    let kitti_dataset_path =
        std::path::Path::new("/Users/kevin/Downloads/2011_09_26/2011_09_26_drive_0087_sync");
    println!("Reading poses...");
    let poses =
        kuba::kitti::oxt_reader::read_from_dir(&kitti_dataset_path.join("oxts/data"), false)
            .unwrap();
    println!("Reading points clouds...");
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
    println!("Opening window...");
    let mut window = kiss3d::window::Window::new("Kitti Point Cloud Vizualizer");
    window.set_light(kiss3d::light::Light::StickToCamera);
    window.set_point_size(1.0);
    let state = AppState {
        poses: poses,
        point_clouds: point_clouds,
        index: 0,
        loop_counter: 0,
    };
    window.render_loop(state);
}
