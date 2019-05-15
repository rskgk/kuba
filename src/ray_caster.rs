use crate::geom;
use crate::geom::Bounds;
use crate::geom::Cell;
use crate::geom::Point;
use crate::geom::PointCloud;

pub fn cast_rays<NaD>(
    origin: &Point<NaD>,
    point_cloud: &PointCloud<NaD>,
    bounds: &Bounds<NaD>,
    resolution: f32,
    max_len: f32,
) -> (
    std::collections::HashSet<Cell<NaD>>,
    std::collections::HashSet<Cell<NaD>>,
)
where
    NaD: na::DimName + std::hash::Hash,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
    <na::DefaultAllocator as na::allocator::Allocator<isize, NaD>>::Buffer: std::hash::Hash,
{
    // TODO(kgreenek): Investigate parallelizing this with rayon.
    let (mut free_cells, occupied_cells) = point_cloud
        .points_iter()
        .map(|point| cast_ray(origin, &point, bounds, resolution, max_len))
        .fold(
            (
                std::collections::HashSet::new(),
                std::collections::HashSet::new(),
            ),
            |(mut free_cells, mut occupied_cells), (ray_cells, end_cell)| {
                if let Some(end_cell) = end_cell {
                    occupied_cells.insert(end_cell);
                }
                for cell in ray_cells {
                    free_cells.insert(cell);
                }
                (free_cells, occupied_cells)
            },
        );
    for cell in &occupied_cells {
        free_cells.remove(cell);
    }
    (free_cells, occupied_cells)
}

/// Returns all the cells that are passed through by the ray from origin to end.
/// The first value of the returned tuple will contain the cells that were cast in the ray. It will
/// NOT include the end cell. The ray will contain the origin unless the ray starts and ends in the
/// same cell; in this case it will be an empty vec.
/// The second value of the returned tuple will contain the end cell if max_len was not exceeded
/// while tracing the ray. If the ray starts and ends in the same cell, it will contain the start
/// cell.
/// The bounds are used to calculate the cells from the given points. However, the returned cells
/// are NOT guaranteed to be contained within the given bounds.
pub fn cast_ray<NaD>(
    origin: &Point<NaD>,
    end: &Point<NaD>,
    bounds: &Bounds<NaD>,
    resolution: f32,
    max_len: f32,
) -> (Vec<Cell<NaD>>, Option<Cell<NaD>>)
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<isize, NaD> + na::allocator::Allocator<f32, NaD>,
{
    assert!(origin.len() == 2 || origin.len() == 3);
    let max_len_sq = max_len * max_len;
    let mut ray_cells = vec![];
    let origin_cell = geom::converter::cell_from_point(&origin, &bounds, resolution);
    let end_cell = geom::converter::cell_from_point(&end, &bounds, resolution);
    if origin_cell == end_cell {
        return (ray_cells, Some(end_cell));
    }
    ray_cells.push(origin_cell.clone());
    let ray = end - origin;
    let ray_length = ray.norm();
    let direction = ray / ray_length;
    let step = geom::vector::signum(&direction);
    let mut current_cell = origin_cell.clone();
    let mut t_max = Point::<NaD>::origin();
    let mut t_delta = Point::<NaD>::origin();
    for i in 0..current_cell.len() {
        if step[i] == 0.0 {
            t_max[i] = std::f32::INFINITY;
            t_delta[i] = std::f32::INFINITY;
            continue;
        }
        let cell_border = geom::converter::point_coord_from_cell_coord(
            current_cell[i],
            bounds.min[i],
            resolution,
        ) + step[i] * resolution * 0.5;
        t_max[i] = (cell_border - origin[i]) / direction[i];
        t_delta[i] = resolution / direction[i].abs();
    }
    loop {
        // This somewhat clunky logic is just a fast way of finding the index of the minimum value
        // in t_max. I.E. which direction (x, y, or z) is the nearest cell border.
        let dim = {
            if origin_cell.len() == 3 {
                if t_max[0] < t_max[1] {
                    if t_max[0] < t_max[2] {
                        0
                    } else {
                        2
                    }
                } else if t_max[1] < t_max[2] {
                    1
                } else {
                    2
                }
            } else {
                if t_max[0] < t_max[1] {
                    0
                } else {
                    1
                }
            }
        };
        current_cell[dim] += step[dim] as isize;
        t_max[dim] += t_delta[dim];
        if current_cell == end_cell {
            break;
        }
        if max_len > 0.0 {
            // Use the squared values to avoid the expensive square root operation to calculate the
            // actual norm.
            let current_ray =
                geom::converter::point_from_cell(&current_cell, &bounds, resolution) - origin;
            if current_ray.norm_squared() > max_len_sq {
                return (ray_cells, None);
            }
        }
        ray_cells.push(current_cell.clone());
    }
    (ray_cells, Some(end_cell))
}

#[cfg(test)]
mod tests {
    use crate as kuba;

    #[test]
    fn cast_rays2() {
        let bounds = kuba::bounds2![[0.0, 0.0], [0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.5;
        let origin = kuba::point2![0.01, 0.0];
        let point_cloud = kuba::PointCloud2::from_points(&[
            kuba::point2![0.31, 0.3],
            kuba::point2![0.3, 0.0],
            kuba::point2![0.0, 0.3],
        ]);
        let (free_cells, occupied_cells) =
            kuba::ray_caster::cast_rays(&origin, &point_cloud, &bounds, resolution, max_len);
        let expected_free_cells: std::collections::HashSet<_> = [
            kuba::cell2![0, 0],
            kuba::cell2![1, 0],
            kuba::cell2![1, 1],
            kuba::cell2![2, 1],
            kuba::cell2![2, 2],
            kuba::cell2![3, 2],
            kuba::cell2![2, 0],
            kuba::cell2![0, 1],
            kuba::cell2![0, 2],
        ].iter().cloned().collect();
        let expected_occupied_cells: std::collections::HashSet<_> = [
            kuba::cell2![3, 3],
            kuba::cell2![0, 3],
            kuba::cell2![3, 0],
        ].iter().cloned().collect();
        assert_eq!(free_cells, expected_free_cells);
        assert_eq!(occupied_cells, expected_occupied_cells);
    }

    #[test]
    fn cast_rays3() {
        let bounds = kuba::bounds3![[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 1.0;
        let origin = kuba::point3![0.01, 0.0, 0.0];
        let point_cloud = kuba::PointCloud3::from_points(&[
            kuba::point3![0.31, 0.3, 0.3],
            kuba::point3![0.3, 0.0, 0.0],
            kuba::point3![0.0, 0.3, 0.0],
            kuba::point3![0.0, 0.0, 0.3],
        ]);
        let (free_cells, occupied_cells) =
            kuba::ray_caster::cast_rays(&origin, &point_cloud, &bounds, resolution, max_len);
        let expected_free_cells: std::collections::HashSet<_> = [
            kuba::cell3![0, 0, 0],
            kuba::cell3![1, 0, 0],
            kuba::cell3![1, 0, 1],
            kuba::cell3![1, 1, 1],
            kuba::cell3![2, 1, 1],
            kuba::cell3![2, 1, 2],
            kuba::cell3![2, 2, 2],
            kuba::cell3![3, 2, 2],
            kuba::cell3![3, 2, 3],
            kuba::cell3![2, 0, 0],
            kuba::cell3![0, 1, 0],
            kuba::cell3![0, 2, 0],
            kuba::cell3![0, 0, 1],
            kuba::cell3![0, 0, 2],
        ].iter().cloned().collect();
        let expected_occupied_cells: std::collections::HashSet<_> = [
            kuba::cell3![3, 3, 3],
            kuba::cell3![3, 0, 0],
            kuba::cell3![0, 3, 0],
            kuba::cell3![0, 0, 3],
        ].iter().cloned().collect();
        assert_eq!(free_cells, expected_free_cells);
        assert_eq!(occupied_cells, expected_occupied_cells);
    }

    #[test]
    fn cast_ray2_diagonal() {
        let bounds = kuba::bounds2![[0.0, 0.0], [0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.5;
        let origin = kuba::point2![0.01, 0.0];
        let end = kuba::point2![0.31, 0.3];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![0, 0],
            kuba::cell2![1, 0],
            kuba::cell2![1, 1],
            kuba::cell2![2, 1],
            kuba::cell2![2, 2],
            kuba::cell2![3, 2],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell2![3, 3]));

        let origin = kuba::point2![0.0, 0.01];
        let end = kuba::point2![0.3, 0.31];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![0, 0],
            kuba::cell2![0, 1],
            kuba::cell2![1, 1],
            kuba::cell2![1, 2],
            kuba::cell2![2, 2],
            kuba::cell2![2, 3],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell2![3, 3]));

        let origin = kuba::point2![0.31, 0.3];
        let end = kuba::point2![0.01, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![3, 3],
            kuba::cell2![3, 2],
            kuba::cell2![2, 2],
            kuba::cell2![2, 1],
            kuba::cell2![1, 1],
            kuba::cell2![1, 0],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell2![0, 0]));

        let origin = kuba::point2![0.31, 0.0];
        let end = kuba::point2![0.0, 0.31];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![3, 0],
            kuba::cell2![2, 0],
            kuba::cell2![2, 1],
            kuba::cell2![1, 1],
            kuba::cell2![1, 2],
            kuba::cell2![0, 2],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell2![0, 3]));

        let origin = kuba::point2![0.0, 0.31];
        let end = kuba::point2![0.31, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![0, 3],
            kuba::cell2![0, 2],
            kuba::cell2![1, 2],
            kuba::cell2![1, 1],
            kuba::cell2![2, 1],
            kuba::cell2![2, 0],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell2![3, 0]));
    }

    #[test]
    fn cast_ray3_diagonal() {
        let bounds = kuba::bounds3![[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 1.0;
        let origin = kuba::point3![0.01, 0.0, 0.0];
        let end = kuba::point3![0.31, 0.3, 0.3];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 0, 0],
            kuba::cell3![1, 0, 0],
            kuba::cell3![1, 0, 1],
            kuba::cell3![1, 1, 1],
            kuba::cell3![2, 1, 1],
            kuba::cell3![2, 1, 2],
            kuba::cell3![2, 2, 2],
            kuba::cell3![3, 2, 2],
            kuba::cell3![3, 2, 3],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![3, 3, 3]));

        let origin = kuba::point3![0.0, 0.01, 0.0];
        let end = kuba::point3![0.3, 0.31, 0.3];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 0, 0],
            kuba::cell3![0, 1, 0],
            kuba::cell3![0, 1, 1],
            kuba::cell3![1, 1, 1],
            kuba::cell3![1, 2, 1],
            kuba::cell3![1, 2, 2],
            kuba::cell3![2, 2, 2],
            kuba::cell3![2, 3, 2],
            kuba::cell3![2, 3, 3],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![3, 3, 3]));

        let origin = kuba::point3![0.31, 0.3, 0.3];
        let end = kuba::point3![0.01, 0.0, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![3, 3, 3],
            kuba::cell3![3, 3, 2],
            kuba::cell3![3, 2, 2],
            kuba::cell3![2, 2, 2],
            kuba::cell3![2, 2, 1],
            kuba::cell3![2, 1, 1],
            kuba::cell3![1, 1, 1],
            kuba::cell3![1, 1, 0],
            kuba::cell3![1, 0, 0],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![0, 0, 0]));

        let origin = kuba::point3![0.31, 0.0, 0.31];
        let end = kuba::point3![0.0, 0.31, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![3, 0, 3],
            kuba::cell3![3, 0, 2],
            kuba::cell3![2, 0, 2],
            kuba::cell3![2, 1, 2],
            kuba::cell3![2, 1, 1],
            kuba::cell3![1, 1, 1],
            kuba::cell3![1, 2, 1],
            kuba::cell3![1, 2, 0],
            kuba::cell3![0, 2, 0],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![0, 3, 0]));

        let origin = kuba::point3![0.0, 0.31, 0.0];
        let end = kuba::point3![0.31, 0.0, 0.31];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 3, 0],
            kuba::cell3![0, 2, 0],
            kuba::cell3![0, 2, 1],
            kuba::cell3![1, 2, 1],
            kuba::cell3![1, 1, 1],
            kuba::cell3![1, 1, 2],
            kuba::cell3![2, 1, 2],
            kuba::cell3![2, 0, 2],
            kuba::cell3![2, 0, 3],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![3, 0, 3]));
    }

    #[test]
    fn cast_ray2_on_axis() {
        let bounds = kuba::bounds2![[0.0, 0.0], [0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.5;
        let origin = kuba::point2![0.0, 0.0];
        let end = kuba::point2![0.3, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![kuba::cell2![0, 0], kuba::cell2![1, 0], kuba::cell2![2, 0]];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell2![3, 0]));

        let origin = kuba::point2![0.3, 0.0];
        let end = kuba::point2![0.0, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![kuba::cell2![3, 0], kuba::cell2![2, 0], kuba::cell2![1, 0]];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell2![0, 0]));

        let origin = kuba::point2![0.0, 0.0];
        let end = kuba::point2![0.0, 0.3];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![kuba::cell2![0, 0], kuba::cell2![0, 1], kuba::cell2![0, 2]];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell2![0, 3]));

        let origin = kuba::point2![0.0, 0.3];
        let end = kuba::point2![0.0, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![kuba::cell2![0, 3], kuba::cell2![0, 2], kuba::cell2![0, 1]];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell2![0, 0]));
    }

    #[test]
    fn cast_ray3_on_axis() {
        let bounds = kuba::bounds3![[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 1.0;
        let origin = kuba::point3![0.0, 0.0, 0.0];
        let end = kuba::point3![0.3, 0.0, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 0, 0],
            kuba::cell3![1, 0, 0],
            kuba::cell3![2, 0, 0],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![3, 0, 0]));

        let origin = kuba::point3![0.3, 0.0, 0.0];
        let end = kuba::point3![0.0, 0.0, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![3, 0, 0],
            kuba::cell3![2, 0, 0],
            kuba::cell3![1, 0, 0],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![0, 0, 0]));

        let origin = kuba::point3![0.0, 0.0, 0.0];
        let end = kuba::point3![0.0, 0.3, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 0, 0],
            kuba::cell3![0, 1, 0],
            kuba::cell3![0, 2, 0],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![0, 3, 0]));

        let origin = kuba::point3![0.0, 0.3, 0.0];
        let end = kuba::point3![0.0, 0.0, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 3, 0],
            kuba::cell3![0, 2, 0],
            kuba::cell3![0, 1, 0],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![0, 0, 0]));

        let origin = kuba::point3![0.0, 0.0, 0.0];
        let end = kuba::point3![0.0, 0.0, 0.3];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 0, 0],
            kuba::cell3![0, 0, 1],
            kuba::cell3![0, 0, 2],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![0, 0, 3]));

        let origin = kuba::point3![0.0, 0.0, 0.3];
        let end = kuba::point3![0.0, 0.0, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 0, 3],
            kuba::cell3![0, 0, 2],
            kuba::cell3![0, 0, 1],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![0, 0, 0]));
    }

    #[test]
    fn cast_ray2_max_len() {
        let bounds = kuba::bounds2![[0.0, 0.0], [0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.42;
        let origin = kuba::point2![0.01, 0.0];
        let end = kuba::point2![0.31, 0.3];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![0, 0],
            kuba::cell2![1, 0],
            kuba::cell2![1, 1],
            kuba::cell2![2, 1],
            kuba::cell2![2, 2],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, None);
    }

    #[test]
    fn cast_ray3_max_len() {
        let bounds = kuba::bounds3![[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.54;
        let origin = kuba::point3![0.01, 0.0, 0.0];
        let end = kuba::point3![0.31, 0.3, 0.3];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 0, 0],
            kuba::cell3![1, 0, 0],
            kuba::cell3![1, 0, 1],
            kuba::cell3![1, 1, 1],
            kuba::cell3![2, 1, 1],
            kuba::cell3![2, 1, 2],
            kuba::cell3![2, 2, 2],
            kuba::cell3![3, 2, 2],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, None);
    }

    #[test]
    fn cast_ray2_with_bounds_offset() {
        let bounds = kuba::bounds2![[-0.1, -0.1], [0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.5;
        let origin = kuba::point2![0.01, 0.0];
        let end = kuba::point2![0.31, 0.3];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![1, 1],
            kuba::cell2![2, 1],
            kuba::cell2![2, 2],
            kuba::cell2![3, 2],
            kuba::cell2![3, 3],
            kuba::cell2![4, 3],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell2![4, 4]));
    }

    #[test]
    fn cast_ray3_with_bounds_offset() {
        let bounds = kuba::bounds3![[-0.1, -0.1, -0.1], [0.4, 0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 1.0;
        let origin = kuba::point3![0.01, 0.0, 0.0];
        let end = kuba::point3![0.31, 0.3, 0.3];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![1, 1, 1],
            kuba::cell3![2, 1, 1],
            kuba::cell3![2, 1, 2],
            kuba::cell3![2, 2, 2],
            kuba::cell3![3, 2, 2],
            kuba::cell3![3, 2, 3],
            kuba::cell3![3, 3, 3],
            kuba::cell3![4, 3, 3],
            kuba::cell3![4, 3, 4],
        ];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![4, 4, 4]));
    }

    #[test]
    fn cast_ray2_origin_same_cell_as_end() {
        let bounds = kuba::bounds2![[0.0, 0.0], [0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.5;
        let origin = kuba::point2![0.01, 0.0];
        let end = kuba::point2![0.02, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell2![0, 0]));
    }

    #[test]
    fn cast_ray3_origin_same_cell_as_end() {
        let bounds = kuba::bounds3![[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 1.0;
        let origin = kuba::point3![0.01, 0.0, 0.0];
        let end = kuba::point3![0.02, 0.0, 0.0];
        let (ray_cells, end_cell) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![];
        assert_eq!(ray_cells, expected_cells);
        assert_eq!(end_cell, Some(kuba::cell3![0, 0, 0]));
    }
}
