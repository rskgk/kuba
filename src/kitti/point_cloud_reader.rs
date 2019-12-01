use byteorder::ReadBytesExt;

use crate::geom::PointCloud3;
use crate::kitti;

pub fn read(input: &mut dyn std::io::Read) -> std::io::Result<PointCloud3> {
    // Kitti points clouds are stored as f32 binary blobs where each point is sequentially stored as
    // [x, y, z, intensity]. We just discard the intensity values.
    // Read the whole file into memory and parse it into a matrix in as one chunk of memory,
    // because this significantly speeds up the parsing.
    let mut point_cloud_bytes = vec![];
    input.read_to_end(&mut point_cloud_bytes)?;
    if point_cloud_bytes.len() % (4 * std::mem::size_of::<f32>()) != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid data length",
        ));
    }
    let point_cloud_buf_size = point_cloud_bytes.len() / std::mem::size_of::<f32>();
    let num_points = point_cloud_buf_size / 4;
    let mut point_cloud_buf = vec![0.0 as f32; point_cloud_buf_size];
    point_cloud_bytes
        .as_slice()
        .read_f32_into::<byteorder::NativeEndian>(&mut point_cloud_buf)?;
    Ok(PointCloud3::from_data(na::MatrixMN::<
        f32,
        na::U3,
        na::Dynamic,
    >::from_iterator(
        num_points,
        point_cloud_buf
            .into_iter()
            .enumerate()
            .filter_map(|(i, val)| if (i + 1) % 4 != 0 { Some(val) } else { None }),
    )))
}

pub fn read_from_file(path: &std::path::Path) -> std::io::Result<PointCloud3> {
    read(&mut std::fs::File::open(path)?)
}

pub fn read_from_dir(
    dir: &std::path::Path,
    print_status: bool,
) -> std::io::Result<Vec<PointCloud3>> {
    let paths = kitti::fs::seq_files_in_dir(dir)?;
    let num_paths = paths.len();
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
            read_from_file(&path)
        })
        .collect()
}
