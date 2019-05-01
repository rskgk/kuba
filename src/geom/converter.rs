use super::*;

/// Returns the point coord at the center of the given cell coord.
/// We use the center of the cell instead of the top left corner to avoid issues with floating
/// point rounding.
#[inline]
pub fn point_coord_from_cell_coord(cell_coord: isize, bounds_min: f32, resolution: f32) -> f32 {
    (cell_coord as f32) * resolution + bounds_min + resolution * 0.5
}

/// Returns the cell coord corresponding to the given point coord.
/// If the point lies exactly on a cell boundary, the higher cell is returned.
#[inline]
pub fn cell_coord_from_point_coord(point_coord: f32, bounds_min: f32, resolution: f32) -> isize {
    let cell_coord = (point_coord - bounds_min) / resolution;
    if approx::relative_eq!(
        cell_coord,
        cell_coord.round(),
        epsilon = std::f32::EPSILON * 10.0
    ) {
        return (cell_coord + resolution * 0.5) as isize;
    }
    cell_coord as isize
}

/// Returns the point at the center of the given cell.
/// We use the center of the cell instead of the top left corner to avoid issues with floating
/// point rounding.
#[inline]
pub fn point_from_cell<NaD>(cell: &Cell<NaD>, bounds: &Bounds<NaD>, resolution: f32) -> Point<NaD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<isize, NaD> + na::allocator::Allocator<f32, NaD>,
{
    Point::<NaD>::from(cell.coords.zip_map(&bounds.min.coords, |coord, min| {
        point_coord_from_cell_coord(coord, min, resolution)
    }))
}

/// Returns the cell corresponding to the given point.
/// If the point lies exactly on a cell boundary, the higher cell is returned.
#[inline]
pub fn cell_from_point<NaD>(point: &Point<NaD>, bounds: &Bounds<NaD>, resolution: f32) -> Cell<NaD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
{
    Cell::<NaD>::from(point.coords.zip_map(&bounds.min.coords, |coord, min| {
        cell_coord_from_point_coord(coord, min, resolution)
    }))
}

#[cfg(test)]
mod tests {
    use crate as kuba;

    #[test]
    fn cell_from_point2() {
        let resolution = 0.1;
        let bounds = kuba::bounds2![[-1.0, -2.0], [9.0, 8.0]];
        for i in 0..100 {
            let point_val = (i as f32) * 0.1;
            assert_eq!(
                kuba::geom::converter::cell_from_point(
                    &kuba::point2![point_val - 0.95, point_val - 1.95],
                    &bounds,
                    resolution
                ),
                kuba::cell2![i, i]
            );
            assert_eq!(
                kuba::geom::converter::cell_from_point(
                    &kuba::point2![point_val - 1.0, point_val - 2.0],
                    &bounds,
                    resolution
                ),
                kuba::cell2![i, i]
            );
        }
    }

    #[test]
    fn cell_from_point3() {
        let resolution = 0.1;
        let bounds = kuba::bounds3![[-1.0, -2.0, -3.0], [9.0, 8.0, 7.0]];
        for i in 0..100 {
            let point_val = (i as f32) * 0.1;
            assert_eq!(
                kuba::geom::converter::cell_from_point(
                    &kuba::point3![point_val - 0.95, point_val - 1.95, point_val - 2.95],
                    &bounds,
                    resolution
                ),
                kuba::cell3![i, i, i]
            );
            assert_eq!(
                kuba::geom::converter::cell_from_point(
                    &kuba::point3![point_val - 1.0, point_val - 2.0, point_val - 3.0],
                    &bounds,
                    resolution
                ),
                kuba::cell3![i, i, i]
            );
        }
    }

    #[test]
    fn point_from_cell2() {
        let resolution = 0.1;
        let bounds = kuba::bounds2![[-1.0, -1.0], [9.0, 9.0]];
        for i in 0..100 {
            let point_val = (i as f32) * 0.1 - 0.95;
            assert!(approx::relative_eq!(
                kuba::geom::converter::point_from_cell(&kuba::cell2![i, i], &bounds, resolution),
                kuba::point2![point_val, point_val]
            ));
        }
    }

    #[test]
    fn point_from_cell3() {
        let resolution = 0.1;
        let bounds = kuba::bounds3![[-1.0, -2.0, -3.0], [9.0, 8.0, 7.0]];
        for i in 0..100 {
            let point_val = (i as f32) * 0.1;
            assert!(approx::relative_eq!(
                kuba::geom::converter::point_from_cell(&kuba::cell3![i, i, i], &bounds, resolution),
                kuba::point3![point_val - 0.95, point_val - 1.95, point_val - 2.95]
            ));
        }
    }
}
