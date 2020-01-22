extern crate kiss3d;
extern crate kuba;
extern crate nalgebra as na;

use kuba::GridMap;
use std::env;
use std::sync::atomic;

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

struct GridUpdater {
    grid_map: std::sync::Mutex<kuba::LidarOccupancyGridMap3>,
    changed_cells: std::sync::Mutex<Vec<kuba::Cell<na::U3>>>,
    finished: atomic::AtomicBool,
}

impl GridUpdater {
    fn new() -> GridUpdater {
        GridUpdater {
            grid_map: std::sync::Mutex::new(kuba::LidarOccupancyGridMap3::default()),
            changed_cells: std::sync::Mutex::new(vec![]),
            finished: atomic::AtomicBool::new(true),
        }
    }
}

struct AppState {
    poses: Vec<kuba::Pose3>,
    point_clouds: Vec<kuba::PointCloud3>,
    index: usize,
    grid_updater: std::sync::Arc<GridUpdater>,
    tracked_cells: std::collections::HashMap<kuba::Cell<na::U3>, na::Point3<f32>>,
}

impl AppState {
    fn integrate_next_pointcloud(&mut self) {
        if !self.grid_updater.finished.load(atomic::Ordering::Relaxed) {
            return;
        }
        {
            let mut changed_cells = self.grid_updater.changed_cells.lock().unwrap();
            let grid_map = self.grid_updater.grid_map.lock().unwrap();
            for cell in changed_cells.iter() {
                let occupied = grid_map.occupied(cell);
                if occupied {
                    if !self.tracked_cells.contains_key(cell) {
                        self.tracked_cells
                            .insert(*cell, grid_map.point_from_cell(cell));
                    }
                } else {
                    if self.tracked_cells.contains_key(cell) {
                        self.tracked_cells.remove(cell);
                    }
                }
            }
            changed_cells.clear();
        }
        self.grid_updater
            .finished
            .store(false, atomic::Ordering::Relaxed);
        let grid_updater = self.grid_updater.clone();
        let origin = kuba::Point3::from(self.poses[self.index].translation.vector);
        let point_cloud = self.point_clouds[self.index].clone();
        std::thread::spawn(move || {
            let mut grid_map = grid_updater.grid_map.lock().unwrap();
            grid_map.set_track_changes(true);
            grid_map.integrate_point_cloud(&origin, &point_cloud);
            {
                let mut changed_cells = grid_updater.changed_cells.lock().unwrap();
                *changed_cells = grid_map.changed_cells();
            }
            grid_map.clear_changed_cells();
            grid_map.set_track_changes(false);
            grid_updater.finished.store(true, atomic::Ordering::Relaxed);
        });
        self.index = (self.index + 1) % self.point_clouds.len();
    }
}

impl kiss3d::window::State for AppState {
    fn step(&mut self, window: &mut kiss3d::window::Window) {
        self.integrate_next_pointcloud();
        let color = na::Point3::new(0.0, 0.6, 0.8);
        for (_, point) in &self.tracked_cells {
            window.draw_point(&point, &color);
        }
        draw_frame_marker(window, &self.poses[self.index], 1.0);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = std::path::Path::new(&args[1]);
    if !path.exists() || !path.is_dir() {
        println!("Invalid source folder");
        return;
    }
    println!("Reading poses...");
    let poses = kuba::kitti::oxt_reader::read_from_dir(&path.join("oxts/data"), false).unwrap();
    println!("Reading point clouds...");
    let point_clouds =
        kuba::kitti::point_cloud_reader::read_from_dir(&path.join("velodyne_points/data"), false)
            .unwrap();
    assert!(point_clouds.len() == poses.len());
    let point_clouds = point_clouds
        .into_iter()
        .zip(&poses)
        .map(|(point_cloud, pose)| point_cloud.transform(&pose.to_homogeneous()))
        .collect();

    println!("Opening window...");
    let mut window = kiss3d::window::Window::new("Kuba Vizualizer");
    window.set_light(kiss3d::light::Light::StickToCamera);
    window.set_point_size(1.0);
    let state = AppState {
        poses: poses,
        point_clouds: point_clouds,
        index: 0,
        grid_updater: std::sync::Arc::new(GridUpdater::new()),
        tracked_cells: std::collections::HashMap::new(),
    };
    window.render_loop(state);
}
