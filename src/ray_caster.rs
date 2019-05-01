use crate::geom;
use crate::geom::Bounds;
use crate::geom::Cell;
use crate::geom::Point;

/// Returns all the cells that are passed through by the ray from origin to end.
/// The ray returned will NOT include the end cell. The ray will contain the origin unless the
/// ray starts and stops in the same cell; in this case an empty vec is returned.
/// The bool returned as the second arg will be true if the end point was reached. It will be false
/// if the max_len was exceeded while tracing the ray.
/// The bounds are used to calculate the cells from the given points. However, the returned cells
/// are NOT guaranteed to be contained within the given bounds.
pub fn cast_ray<NaD>(
    origin: &Point<NaD>,
    end: &Point<NaD>,
    bounds: &Bounds<NaD>,
    resolution: f32,
    max_len: f32,
) -> (Vec<Cell<NaD>>, bool)
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
        return (ray_cells, true);
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
        let mut dim = 2;
        // This is a bit clunky. Maybe there's a way to handle 2 or 3 dimensions more gracefully?
        if origin_cell.len() == 3 {
            if t_max[0] < t_max[1] {
                if t_max[0] < t_max[2] {
                    dim = 0;
                }
            } else if t_max[1] < t_max[2] {
                dim = 1;
            }
        } else {
            if t_max[0] < t_max[1] {
                dim = 0;
            } else {
                dim = 1;
            }
        }
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
                return (ray_cells, false);
            }
        }
        ray_cells.push(current_cell.clone());
    }
    (ray_cells, true)
}

#[cfg(test)]
mod tests {
    use crate as kuba;

    #[test]
    fn cast_ray2_diagonal() {
        let bounds = kuba::bounds2![[0.0, 0.0], [0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.5;
        let origin = kuba::point2![0.01, 0.0];
        let end = kuba::point2![0.31, 0.3];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![0, 0],
            kuba::cell2![1, 0],
            kuba::cell2![1, 1],
            kuba::cell2![2, 1],
            kuba::cell2![2, 2],
            kuba::cell2![3, 2],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point2![0.0, 0.01];
        let end = kuba::point2![0.3, 0.31];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![0, 0],
            kuba::cell2![0, 1],
            kuba::cell2![1, 1],
            kuba::cell2![1, 2],
            kuba::cell2![2, 2],
            kuba::cell2![2, 3],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point2![0.31, 0.3];
        let end = kuba::point2![0.01, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![3, 3],
            kuba::cell2![3, 2],
            kuba::cell2![2, 2],
            kuba::cell2![2, 1],
            kuba::cell2![1, 1],
            kuba::cell2![1, 0],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point2![0.31, 0.0];
        let end = kuba::point2![0.0, 0.31];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![3, 0],
            kuba::cell2![2, 0],
            kuba::cell2![2, 1],
            kuba::cell2![1, 1],
            kuba::cell2![1, 2],
            kuba::cell2![0, 2],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point2![0.0, 0.31];
        let end = kuba::point2![0.31, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![0, 3],
            kuba::cell2![0, 2],
            kuba::cell2![1, 2],
            kuba::cell2![1, 1],
            kuba::cell2![2, 1],
            kuba::cell2![2, 0],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);
    }

    #[test]
    fn cast_ray3_diagonal() {
        let bounds = kuba::bounds3![[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 1.0;
        let origin = kuba::point3![0.01, 0.0, 0.0];
        let end = kuba::point3![0.31, 0.3, 0.3];
        let (cast_cells, end_reached) =
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
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point3![0.0, 0.01, 0.0];
        let end = kuba::point3![0.3, 0.31, 0.3];
        let (cast_cells, end_reached) =
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
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point3![0.31, 0.3, 0.3];
        let end = kuba::point3![0.01, 0.0, 0.0];
        let (cast_cells, end_reached) =
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
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point3![0.31, 0.0, 0.31];
        let end = kuba::point3![0.0, 0.31, 0.0];
        let (cast_cells, end_reached) =
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
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point3![0.0, 0.31, 0.0];
        let end = kuba::point3![0.31, 0.0, 0.31];
        let (cast_cells, end_reached) =
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
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);
    }

    #[test]
    fn cast_ray2_on_axis() {
        let bounds = kuba::bounds2![[0.0, 0.0], [0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.5;
        let origin = kuba::point2![0.0, 0.0];
        let end = kuba::point2![0.3, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![0, 0],
            kuba::cell2![1, 0],
            kuba::cell2![2, 0],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point2![0.3, 0.0];
        let end = kuba::point2![0.0, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![3, 0],
            kuba::cell2![2, 0],
            kuba::cell2![1, 0],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point2![0.0, 0.0];
        let end = kuba::point2![0.0, 0.3];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![0, 0],
            kuba::cell2![0, 1],
            kuba::cell2![0, 2],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point2![0.0, 0.3];
        let end = kuba::point2![0.0, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![0, 3],
            kuba::cell2![0, 2],
            kuba::cell2![0, 1],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);
    }

    #[test]
    fn cast_ray3_on_axis() {
        let bounds = kuba::bounds3![[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 1.0;
        let origin = kuba::point3![0.0, 0.0, 0.0];
        let end = kuba::point3![0.3, 0.0, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 0, 0],
            kuba::cell3![1, 0, 0],
            kuba::cell3![2, 0, 0],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point3![0.3, 0.0, 0.0];
        let end = kuba::point3![0.0, 0.0, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![3, 0, 0],
            kuba::cell3![2, 0, 0],
            kuba::cell3![1, 0, 0],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point3![0.0, 0.0, 0.0];
        let end = kuba::point3![0.0, 0.3, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 0, 0],
            kuba::cell3![0, 1, 0],
            kuba::cell3![0, 2, 0],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point3![0.0, 0.3, 0.0];
        let end = kuba::point3![0.0, 0.0, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 3, 0],
            kuba::cell3![0, 2, 0],
            kuba::cell3![0, 1, 0],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point3![0.0, 0.0, 0.0];
        let end = kuba::point3![0.0, 0.0, 0.3];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 0, 0],
            kuba::cell3![0, 0, 1],
            kuba::cell3![0, 0, 2],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);

        let origin = kuba::point3![0.0, 0.0, 0.3];
        let end = kuba::point3![0.0, 0.0, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell3![0, 0, 3],
            kuba::cell3![0, 0, 2],
            kuba::cell3![0, 0, 1],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);
    }

    #[test]
    fn cast_ray2_max_len() {
        let bounds = kuba::bounds2![[0.0, 0.0], [0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.42;
        let origin = kuba::point2![0.01, 0.0];
        let end = kuba::point2![0.31, 0.3];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![0, 0],
            kuba::cell2![1, 0],
            kuba::cell2![1, 1],
            kuba::cell2![2, 1],
            kuba::cell2![2, 2],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, false);
    }

    #[test]
    fn cast_ray3_max_len() {
        let bounds = kuba::bounds3![[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.54;
        let origin = kuba::point3![0.01, 0.0, 0.0];
        let end = kuba::point3![0.31, 0.3, 0.3];
        let (cast_cells, end_reached) =
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
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, false);
    }

    #[test]
    fn cast_ray2_with_bounds_offset() {
        let bounds = kuba::bounds2![[-0.1, -0.1], [0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.5;
        let origin = kuba::point2![0.01, 0.0];
        let end = kuba::point2![0.31, 0.3];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![
            kuba::cell2![1, 1],
            kuba::cell2![2, 1],
            kuba::cell2![2, 2],
            kuba::cell2![3, 2],
            kuba::cell2![3, 3],
            kuba::cell2![4, 3],
        ];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);
    }

    #[test]
    fn cast_ray3_with_bounds_offset() {
        let bounds = kuba::bounds3![[-0.1, -0.1, -0.1], [0.4, 0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 1.0;
        let origin = kuba::point3![0.01, 0.0, 0.0];
        let end = kuba::point3![0.31, 0.3, 0.3];
        let (cast_cells, end_reached) =
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
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);
    }

    #[test]
    fn cast_ray2_origin_same_cell_as_end() {
        let bounds = kuba::bounds2![[0.0, 0.0], [0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 0.5;
        let origin = kuba::point2![0.01, 0.0];
        let end = kuba::point2![0.02, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);
    }

    #[test]
    fn cast_ray3_origin_same_cell_as_end() {
        let bounds = kuba::bounds3![[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]];
        let resolution = 0.1;
        let max_len = 1.0;
        let origin = kuba::point3![0.01, 0.0, 0.0];
        let end = kuba::point3![0.02, 0.0, 0.0];
        let (cast_cells, end_reached) =
            kuba::ray_caster::cast_ray(&origin, &end, &bounds, resolution, max_len);
        let expected_cells = vec![];
        assert_eq!(cast_cells, expected_cells);
        assert_eq!(end_reached, true);
    }
}
