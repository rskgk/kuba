extern crate kiss3d;
extern crate kuba;
extern crate nalgebra as na;

fn draw_frame_marker(window: &mut kiss3d::window::Window, origin: &na::Point3<f32>, length: f32) {
    window.draw_line(
        origin,
        &na::Point3::new(length, 0.0, 0.0),
        &na::Point3::new(1.0, 0.0, 0.0),
    );
    window.draw_line(
        origin,
        &na::Point3::new(0.0, length, 0.0),
        &na::Point3::new(0.0, 1.0, 0.0),
    );
    window.draw_line(
        origin,
        &na::Point3::new(0.0, 0.0, length),
        &na::Point3::new(0.0, 0.0, 1.0),
    );
}

struct AppState {
    poses: Vec<kuba::Pose3>,
    point_clouds: Vec<kuba::PointCloud3>,
}

impl kiss3d::window::State for AppState {
    fn step(&mut self, window: &mut kiss3d::window::Window) {
        for point in self.point_clouds[0].points_iter() {
            window.draw_point(&point, &na::Point3::new(0.0, 0.6, 0.8));
        }
        draw_frame_marker(window, &na::Point3::origin(), 1.0);
    }
}

fn main() {
    let kitti_dataset_path =
        std::path::Path::new("/Users/kevin/Downloads/2011_09_26/2011_09_26_drive_0087_sync");
    println!("Reading poses...");
    let poses = kuba::kitti::oxt_reader::read_from_dir(
        &kitti_dataset_path.join("oxts/data"),
        false,
    )
    .unwrap();
    println!("Reading points clouds...");
    let point_clouds = kuba::kitti::point_cloud_reader::read_from_dir(
        &kitti_dataset_path.join("velodyne_points/data"),
        false,
    )
    .unwrap();
    println!("Opening window...");
    let mut window = kiss3d::window::Window::new("Kitti Point Cloud Vizualizer");
    window.set_light(kiss3d::light::Light::StickToCamera);
    window.set_point_size(1.0);
    let state = AppState {
        poses: poses,
        point_clouds: point_clouds,
    };
    window.render_loop(state);
}
