use crate::geom;
use crate::geom::Bounds;
use crate::geom::Cell;
use crate::geom::CellToNdIndex;
use crate::geom::Point;

pub type GridMap2<A> = GridMapN<A, na::U2, nd::Ix2>;
pub type GridMap2f = GridMap2<f32>;
pub type GridMap2i = GridMap2<i32>;
pub type GridMap2b = GridMap2<bool>;

pub type GridMap3<A> = GridMapN<A, na::U3, nd::Ix3>;
pub type GridMap3f = GridMap3<f32>;
pub type GridMap3i = GridMap2<i32>;
pub type GridMap3b = GridMap2<bool>;

// TODO(kgreenek): It's annoying to have to expose NaD and NdD. Figure out a way to just have one
// generic dimention parameter.
pub struct GridMapN<A, NaD, NdD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD>,
{
    data: nd::Array<A, NdD>,
    resolution: f32,
    bounds: Bounds<NaD>,
}

impl<A, NaD, NdD> GridMapN<A, NaD, NdD>
where
    A: na::Scalar,
    NaD: na::DimName,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    na::DefaultAllocator: na::allocator::Allocator<A, NaD>
        + na::allocator::Allocator<f32, NaD>
        + na::allocator::Allocator<isize, NaD>,
{
    pub fn from_ndarray(ndarray: nd::Array<A, NdD>, resolution: f32, bounds: Bounds<NaD>) -> Self {
        GridMapN {
            data: ndarray,
            resolution: resolution,
            bounds: bounds,
        }
    }

    pub fn from_bounds(resolution: f32, bounds: &Bounds<NaD>, default_value: A) -> Self {
        let size_cells = geom::converter::cell_from_point(&bounds.max, &bounds.min, resolution);
        let vec_size: isize = size_cells.iter().product();
        let array_vec = std::vec::from_elem(default_value, vec_size as usize);
        let data = nd::Array::<A, NdD>::from_shape_vec(size_cells.to_ndindex(), array_vec).unwrap();
        GridMapN {
            data: data,
            resolution: resolution,
            bounds: bounds.clone(),
        }
    }

    /// Returns the value at the given cell.
    #[inline]
    pub fn get(&self, cell: &Cell<NaD>) -> A {
        self.data[cell.to_ndindex()]
    }

    /// Sets the value at the given cell.
    #[inline]
    pub fn set(&mut self, cell: &Cell<NaD>, value: A) {
        self.data[cell.to_ndindex()] = value;
    }

    /// Returns the point at the center of the given cell.
    /// We use the center of the cell instead of the top left corner to avoid issues with floating
    /// point rounding.
    #[inline]
    pub fn point_from_cell(&self, cell: &Cell<NaD>) -> Point<NaD> {
        geom::converter::point_from_cell(cell, &self.bounds.min, self.resolution)
    }

    /// Returns the cell corresponding to the given point.
    /// If the point lies exactly on a cell boundary, the higher cell is returned.
    #[inline]
    pub fn cell_from_point(&self, point: &Point<NaD>) -> Cell<NaD> {
        geom::converter::cell_from_point(point, &self.bounds.min, self.resolution)
    }
}

// TODO(kgreenek): Support a generic resized method instead of copying it for 2d and 3d.
// In a generic context, you can't use the ns::s![] macro, because it returns a fixed size
// array rather than a <NdD as nd::Dimension>::SliceArg generic type that is required for
// calling self.grid_map.data in a generic context.
impl<A> GridMap2<A>
where
    A: na::Scalar,
{
    pub fn resized(&self, bounds: &Bounds<na::U2>, default_value: A) -> Self {
        let mut grid_map = Self::from_bounds(self.resolution, bounds, default_value);
        let overlapping_bounds = self.bounds.overlapping(bounds);
        if overlapping_bounds == Bounds::<na::U2>::empty() {
            return grid_map;
        }
        let from_cell_min = geom::converter::cell_from_point(
            &overlapping_bounds.min,
            &self.bounds.min,
            self.resolution,
        );
        let from_cell_max = geom::converter::cell_from_point(
            &overlapping_bounds.max,
            &self.bounds.min,
            self.resolution,
        );
        let to_cell_min =
            geom::converter::cell_from_point(&overlapping_bounds.min, &bounds.min, self.resolution);
        let to_cell_max =
            geom::converter::cell_from_point(&overlapping_bounds.max, &bounds.min, self.resolution);

        let mut to_slice = grid_map.data.slice_mut(nd::s![
            to_cell_min.coords[0]..to_cell_max.coords[0],
            to_cell_min.coords[1]..to_cell_max.coords[1]
        ]);
        let from_slice = self.data.slice(nd::s![
            from_cell_min.coords[0]..from_cell_max.coords[0],
            from_cell_min.coords[1]..from_cell_max.coords[1]
        ]);
        to_slice.assign(&from_slice);
        grid_map
    }
}

// TODO(kgreenek): Support a generic resized method instead of copying it for 2d and 3d.
// In a generic context, you can't use the ns::s![] macro, because it returns a fixed size
// array rather than a <NdD as nd::Dimension>::SliceArg generic type that is required for
// calling self.grid_map.data in a generic context.
impl<A> GridMap3<A>
where
    A: na::Scalar,
{
    pub fn resized(&self, bounds: &Bounds<na::U3>, default_value: A) -> Self {
        let mut grid_map = Self::from_bounds(self.resolution, bounds, default_value);
        let overlapping_bounds = self.bounds.overlapping(bounds);
        if overlapping_bounds == Bounds::<na::U3>::empty() {
            return grid_map;
        }
        let from_cell_min = geom::converter::cell_from_point(
            &overlapping_bounds.min,
            &self.bounds.min,
            self.resolution,
        );
        let from_cell_max = geom::converter::cell_from_point(
            &overlapping_bounds.max,
            &self.bounds.min,
            self.resolution,
        );
        let to_cell_min =
            geom::converter::cell_from_point(&overlapping_bounds.min, &bounds.min, self.resolution);
        let to_cell_max =
            geom::converter::cell_from_point(&overlapping_bounds.max, &bounds.min, self.resolution);

        let mut to_slice = grid_map.data.slice_mut(nd::s![
            to_cell_min.coords[0]..to_cell_max.coords[0],
            to_cell_min.coords[1]..to_cell_max.coords[1],
            to_cell_min.coords[2]..to_cell_max.coords[2]
        ]);
        let from_slice = self.data.slice(nd::s![
            from_cell_min.coords[0]..from_cell_max.coords[0],
            from_cell_min.coords[1]..from_cell_max.coords[1],
            from_cell_min.coords[2]..from_cell_max.coords[2]
        ]);
        to_slice.assign(&from_slice);
        grid_map
    }
}

#[cfg(test)]
mod tests {
    use crate as kuba;

    #[test]
    fn get2() {
        let grid_map = kuba::GridMap2f::from_ndarray(
            nd::array![[1.0, 2.0], [10.0, 20.0]],
            0.1,
            kuba::bounds2![[0.0, 0.0], [0.2, 0.2]],
        );
        assert_eq!(grid_map.get(&kuba::cell2![0, 0]), 1.0);
        assert_eq!(grid_map.get(&kuba::cell2![0, 1]), 2.0);
        assert_eq!(grid_map.get(&kuba::cell2![1, 0]), 10.0);
        assert_eq!(grid_map.get(&kuba::cell2![1, 1]), 20.0);
    }

    #[test]
    fn get3() {
        let grid_map = kuba::GridMap3f::from_ndarray(
            nd::array![
                [[111.0, 112.0], [121.0, 122.0]],
                [[211.0, 212.0], [221.0, 222.0]],
            ],
            0.1,
            kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
        );
        assert_eq!(grid_map.get(&kuba::cell3![0, 0, 0]), 111.0);
        assert_eq!(grid_map.get(&kuba::cell3![0, 0, 1]), 112.0);
        assert_eq!(grid_map.get(&kuba::cell3![0, 1, 0]), 121.0);
        assert_eq!(grid_map.get(&kuba::cell3![0, 1, 1]), 122.0);
        assert_eq!(grid_map.get(&kuba::cell3![1, 0, 0]), 211.0);
        assert_eq!(grid_map.get(&kuba::cell3![1, 0, 1]), 212.0);
        assert_eq!(grid_map.get(&kuba::cell3![1, 1, 0]), 221.0);
        assert_eq!(grid_map.get(&kuba::cell3![1, 1, 1]), 222.0);
    }

    #[test]
    fn set2() {
        let mut grid_map = kuba::GridMap2f::from_ndarray(
            nd::array![[1.0, 2.0], [10.0, 20.0]],
            0.1,
            kuba::bounds2![[0.0, 0.0], [0.2, 0.2]],
        );
        grid_map.set(&kuba::cell2![0, 0], 15.0);
        assert_eq!(grid_map.get(&kuba::cell2![0, 0]), 15.0);
    }

    #[test]
    fn set3() {
        let mut grid_map = kuba::GridMap3f::from_ndarray(
            nd::array![
                [[111.0, 112.0], [121.0, 122.0]],
                [[211.0, 212.0], [221.0, 222.0]],
            ],
            0.1,
            kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
        );
        grid_map.set(&kuba::cell3![0, 0, 0], 15.0);
        grid_map.set(&kuba::cell3![0, 1, 0], 25.0);
        assert_eq!(grid_map.get(&kuba::cell3![0, 0, 0]), 15.0);
        assert_eq!(grid_map.get(&kuba::cell3![0, 1, 0]), 25.0);
    }

    #[test]
    fn resized2_nominal() {
        let grid_map =
            kuba::GridMap2f::from_bounds(0.1, &kuba::bounds2![[0.0, 0.0], [0.2, 0.2]], 1.0);
        let resized_grid_map = grid_map.resized(&kuba::bounds2![[0.1, 0.1], [0.3, 0.3]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 1]), 0.0);
        let resized_grid_map = grid_map.resized(&kuba::bounds2![[-0.1, -0.1], [0.1, 0.1]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 1]), 1.0);
        let resized_grid_map = grid_map.resized(&kuba::bounds2![[-0.1, 0.1], [0.1, 0.3]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 1]), 0.0);
        let resized_grid_map = grid_map.resized(&kuba::bounds2![[0.1, -0.1], [0.3, 0.1]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 1]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 1]), 0.0);
    }

    #[test]
    fn resized3_nominal() {
        let grid_map = kuba::GridMap3f::from_bounds(
            0.1,
            &kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
            1.0,
        );
        let resized_grid_map =
            grid_map.resized(&kuba::bounds3![[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 1]), 0.0);
        let resized_grid_map =
            grid_map.resized(&kuba::bounds3![[0.1, 0.1, -0.1], [0.3, 0.3, 0.1]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 1]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 1]), 0.0);
        let resized_grid_map =
            grid_map.resized(&kuba::bounds3![[0.1, -0.1, 0.1], [0.3, 0.1, 0.3]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 1]), 0.0);
        let resized_grid_map =
            grid_map.resized(&kuba::bounds3![[0.1, -0.1, -0.1], [0.3, 0.1, 0.1]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 1]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 1]), 0.0);
        let resized_grid_map =
            grid_map.resized(&kuba::bounds3![[-0.1, 0.1, 0.1], [0.1, 0.3, 0.3]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 1]), 0.0);
        let resized_grid_map =
            grid_map.resized(&kuba::bounds3![[-0.1, 0.1, -0.1], [0.1, 0.3, 0.1]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 1]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 1]), 0.0);
        let resized_grid_map =
            grid_map.resized(&kuba::bounds3![[-0.1, -0.1, 0.1], [0.1, 0.1, 0.3]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 1]), 0.0);
        let resized_grid_map =
            grid_map.resized(&kuba::bounds3![[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 1]), 1.0);
    }

    #[test]
    fn resized2_no_overlap() {
        let grid_map =
            kuba::GridMap2f::from_bounds(0.1, &kuba::bounds2![[0.0, 0.0], [0.2, 0.2]], 1.0);
        let resized_grid_map = grid_map.resized(&kuba::bounds2![[0.3, 0.3], [0.5, 0.5]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 1]), 0.0);
    }

    #[test]
    fn resized3_no_overlap() {
        let grid_map = kuba::GridMap3f::from_bounds(
            0.1,
            &kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
            1.0,
        );
        let resized_grid_map =
            grid_map.resized(&kuba::bounds3![[0.3, 0.3, 0.3], [0.5, 0.5, 0.5]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 1]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 0]), 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 1]), 0.0);
    }

    #[test]
    fn resized2_all_overlap() {
        let grid_map =
            kuba::GridMap2f::from_bounds(0.1, &kuba::bounds2![[0.0, 0.0], [0.2, 0.2]], 1.0);
        let resized_grid_map = grid_map.resized(&kuba::bounds2![[0.0, 0.0], [0.2, 0.2]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![0, 1]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell2![1, 1]), 1.0);
    }

    #[test]
    fn resized3_all_overlap() {
        let grid_map = kuba::GridMap3f::from_bounds(
            0.1,
            &kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
            1.0,
        );
        let resized_grid_map =
            grid_map.resized(&kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]], 0.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 0, 1]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![0, 1, 1]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 0, 1]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 0]), 1.0);
        assert_eq!(resized_grid_map.get(&kuba::cell3![1, 1, 1]), 1.0);
    }
}
