use crate::geom::Pose3;
use crate::kitti;

/// The circumference of the earth along the equator.
const EARTH_EQUATOR_CIRCUMFERENCE: f32 = 40075160.0;
/// The circumference of the earth traveling through the poles.
const EARTH_POLE_CIRCUMFERENCE: f32 = 40008000.0;

fn str_to_f32(string: &str) -> std::io::Result<f32> {
    match string.parse::<f32>() {
        Ok(val) => Ok(val),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid f32",
        )),
    }
}

/// Converts the given gps_point to a point in meters, relative to the given gps_origin.
/// NOTE: This is just an approximation. It will not be accurate if the distances are large enough
/// that the curvature of the earth makes a significant difference.
/// Longitude is treated as positive x, and latitude is treated as positive y.
/// The z height is calculated as the difference in altitude from the gps_origin.
/// See:
///   https://stackoverflow.com/questions/3024404/transform-longitude-latitude-into-meters
fn gps_to_meters(
    gps_point: &na::Translation3<f32>,
    gps_origin: &na::Translation3<f32>,
) -> na::Translation3<f32> {
    let dlat = gps_point.vector[0] - gps_origin.vector[0];
    let dlon = gps_point.vector[1] - gps_origin.vector[1];
    let lat_circumference = EARTH_EQUATOR_CIRCUMFERENCE * dlat.to_radians().cos();
    na::Translation3::<f32>::new(
        dlon * lat_circumference / 360.0,
        dlat * EARTH_POLE_CIRCUMFERENCE / 360.0,
        gps_point.vector[2] - gps_origin.vector[2],
    )
}

pub fn read(
    input: &mut std::io::Read,
    gps_origin: Option<na::Translation3<f32>>,
) -> std::io::Result<(Pose3, na::Translation3<f32>)> {
    let mut buffer = String::new();
    input.read_to_string(&mut buffer)?;
    let str_values = buffer.split(" ").collect::<Vec<&str>>();
    let err = Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "Invalid data length",
    ));
    if str_values.len() < 6 {
        return err;
    }
    let gps_point = na::Translation3::<f32>::new(
        str_to_f32(str_values[0])?,
        str_to_f32(str_values[1])?,
        str_to_f32(str_values[2])?,
    );
    let rotation = na::UnitQuaternion::<f32>::from_euler_angles(
        str_to_f32(str_values[3])?,
        str_to_f32(str_values[4])?,
        str_to_f32(str_values[5])?,
    );
    if let Some(gps_origin) = gps_origin {
        Ok((
            Pose3::from_parts(gps_to_meters(&gps_point, &gps_origin), rotation),
            gps_origin,
        ))
    } else {
        Ok((
            Pose3::from_parts(na::Translation3::<f32>::new(0.0, 0.0, 0.0), rotation),
            gps_point,
        ))
    }
}

pub fn read_from_file(
    path: &std::path::Path,
    gps_origin: Option<na::Translation3<f32>>,
) -> std::io::Result<(Pose3, na::Translation3<f32>)> {
    read(&mut std::fs::File::open(path)?, gps_origin)
}

pub fn read_from_dir(dir: &std::path::Path, print_status: bool) -> std::io::Result<Vec<Pose3>> {
    let paths = kitti::fs::seq_files_in_dir(dir)?;
    let num_paths = paths.len();
    let mut gps_origin = None;
    paths
        .into_iter()
        .enumerate()
        .map(|(i, path)| {
            if print_status {
                println!(
                    "Reading file ({} of {}) {}",
                    i + 1,
                    num_paths,
                    path.file_name().unwrap().to_str().unwrap()
                );
            }
            match read_from_file(&path, gps_origin) {
                Ok((pose, new_gps_origin)) => {
                    gps_origin = Some(new_gps_origin);
                    Ok(pose)
                },
                Err(err) => Err(err)
            }
        })
        .collect()
}
