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

pub trait GridMap<A, NaD, NdD>
where
    A: na::Scalar,
    NaD: na::DimName,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    na::DefaultAllocator: na::allocator::Allocator<A, NaD>
        + na::allocator::Allocator<f32, NaD>
        + na::allocator::Allocator<isize, NaD>,
{
    #[inline]
    fn get(&self, cell: &Cell<NaD>) -> A;

    #[inline]
    fn set(&mut self, cell: &Cell<NaD>, value: A);

    #[inline]
    fn resolution(&self) -> f32;

    #[inline]
    fn bounds(&self) -> Bounds<NaD>;

    /// Returns the point at the center of the given cell.
    /// We use the center of the cell instead of the top left corner to avoid issues with floating
    /// point rounding.
    #[inline]
    fn point_from_cell(&self, cell: &Cell<NaD>) -> Point<NaD>;

    /// Returns the cell corresponding to the given point.
    /// If the point lies exactly on a cell boundary, the higher cell is returned.
    #[inline]
    fn cell_from_point(&self, point: &Point<NaD>) -> Cell<NaD>;
}

pub trait ExpandableGridMap<A, NaD, NdD>: GridMap<A, NaD, NdD>
where
    A: na::Scalar,
    NaD: na::DimName,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    na::DefaultAllocator: na::allocator::Allocator<A, NaD>
        + na::allocator::Allocator<f32, NaD>
        + na::allocator::Allocator<isize, NaD>,
{
    /// Expand the bounds to include the given bounds. Ensures the memory is allocated to set the
    /// values in the grid_map within the given bounds.
    fn expand_bounds(&mut self, bounds: &Bounds<NaD>, default_value: A);
}

pub trait ResizableGridMap<A, NaD, NdD>: GridMap<A, NaD, NdD>
where
    A: na::Scalar,
    NaD: na::DimName,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    na::DefaultAllocator: na::allocator::Allocator<A, NaD>
        + na::allocator::Allocator<f32, NaD>
        + na::allocator::Allocator<isize, NaD>,
{
    fn resize(&mut self, bounds: &Bounds<NaD>, default_value: A);
    fn resized(&self, bounds: &Bounds<NaD>, default_value: A) -> Self;
}

// TODO(kgreenek): It's annoying to have to expose NaD and NdD. Figure out a way to just have one
// generic dimention parameter.
pub struct GridMapN<A, NaD, NdD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD>,
{
    pub data: nd::Array<A, NdD>,
    pub resolution: f32,
    pub bounds: Bounds<NaD>,
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

    pub fn from_bounds(resolution: f32, bounds: Bounds<NaD>, default_value: A) -> Self {
        GridMapN {
            data: ndarray_with_bounds(resolution, &bounds, default_value),
            resolution: resolution,
            bounds: bounds.clone(),
        }
    }
}

impl<A, NaD, NdD> GridMap<A, NaD, NdD> for GridMapN<A, NaD, NdD>
where
    A: na::Scalar,
    NaD: na::DimName,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    na::DefaultAllocator: na::allocator::Allocator<A, NaD>
        + na::allocator::Allocator<f32, NaD>
        + na::allocator::Allocator<isize, NaD>,
{
    #[inline]
    fn get(&self, cell: &Cell<NaD>) -> A {
        self.data[cell.to_ndindex()]
    }

    #[inline]
    fn set(&mut self, cell: &Cell<NaD>, value: A) {
        self.data[cell.to_ndindex()] = value;
    }

    #[inline]
    fn resolution(&self) -> f32 {
        self.resolution
    }

    #[inline]
    fn bounds(&self) -> Bounds<NaD> {
        self.bounds.clone()
    }

    #[inline]
    fn point_from_cell(&self, cell: &Cell<NaD>) -> Point<NaD> {
        geom::converter::point_from_cell(cell, &self.bounds.min, self.resolution)
    }

    #[inline]
    fn cell_from_point(&self, point: &Point<NaD>) -> Cell<NaD> {
        geom::converter::cell_from_point(point, &self.bounds.min, self.resolution)
    }
}

// TODO(kgreenek): Support a generic resized method instead of copying it for 2d and 3d.
// In a generic context, you can't use the ns::s![] macro, because it returns a fixed size
// array rather than a <NdD as nd::Dimension>::SliceArg generic type that is required for
// calling self.grid_map.data in a generic context.
impl<A> ExpandableGridMap<A, na::U2, nd::Ix2> for GridMapN<A, na::U2, nd::Ix2>
where
    A: na::Scalar,
{
    fn expand_bounds(&mut self, bounds: &Bounds<na::U2>, default_value: A) {
        // Expand the bounds here when discretizing in case any point in the point_cloud lie exactly
        // on a cell boundary. Otherwise it won't be resized to include the cell containing the
        // boundary point.
        let new_bounds = self
            .bounds
            .enclosing(&bounds.discretized(self.resolution, true));
        self.data = resized_ndarray2(
            &self.data,
            &self.bounds,
            &new_bounds,
            self.resolution,
            default_value,
        );
        self.bounds = new_bounds;
    }
}

// TODO(kgreenek): Support a generic resized method instead of copying it for 2d and 3d.
// In a generic context, you can't use the ns::s![] macro, because it returns a fixed size
// array rather than a <NdD as nd::Dimension>::SliceArg generic type that is required for
// calling self.grid_map.data in a generic context.
impl<A> ExpandableGridMap<A, na::U3, nd::Ix3> for GridMapN<A, na::U3, nd::Ix3>
where
    A: na::Scalar,
{
    fn expand_bounds(&mut self, bounds: &Bounds<na::U3>, default_value: A) {
        // Expand the bounds here when discretizing in case any point in the point_cloud lie exactly
        // on a cell boundary. Otherwise it won't be resized to include the cell containing the
        // boundary point.
        let new_bounds = self
            .bounds
            .enclosing(&bounds.discretized(self.resolution, true));
        self.data = resized_ndarray3(
            &self.data,
            &self.bounds,
            &new_bounds,
            self.resolution,
            default_value,
        );
        self.bounds = new_bounds;
    }
}

// TODO(kgreenek): Support a generic resized method instead of copying it for 2d and 3d.
// In a generic context, you can't use the ns::s![] macro, because it returns a fixed size
// array rather than a <NdD as nd::Dimension>::SliceArg generic type that is required for
// calling self.grid_map.data in a generic context.
impl<A> ResizableGridMap<A, na::U2, nd::Ix2> for GridMap2<A>
where
    A: na::Scalar,
{
    fn resize(&mut self, bounds: &Bounds<na::U2>, default_value: A) {
        self.data = resized_ndarray2(
            &self.data,
            &self.bounds,
            bounds,
            self.resolution,
            default_value,
        );
        self.bounds = bounds.clone();
    }

    fn resized(&self, bounds: &Bounds<na::U2>, default_value: A) -> GridMap2<A> {
        GridMap2 {
            data: resized_ndarray2(
                &self.data,
                &self.bounds,
                bounds,
                self.resolution,
                default_value,
            ),
            bounds: bounds.clone(),
            resolution: self.resolution,
        }
    }
}

// TODO(kgreenek): Support a generic resized method instead of copying it for 2d and 3d.
// In a generic context, you can't use the ns::s![] macro, because it returns a fixed size
// array rather than a <NdD as nd::Dimension>::SliceArg generic type that is required for
// calling self.grid_map.data in a generic context.
impl<A> ResizableGridMap<A, na::U3, nd::Ix3> for GridMap3<A>
where
    A: na::Scalar,
{
    fn resize(&mut self, bounds: &Bounds<na::U3>, default_value: A) {
        self.data = resized_ndarray3(
            &self.data,
            &self.bounds,
            bounds,
            self.resolution,
            default_value,
        );
        self.bounds = bounds.clone();
    }

    fn resized(&self, bounds: &Bounds<na::U3>, default_value: A) -> GridMap3<A> {
        GridMap3 {
            data: resized_ndarray3(
                &self.data,
                &self.bounds,
                bounds,
                self.resolution,
                default_value,
            ),
            bounds: bounds.clone(),
            resolution: self.resolution,
        }
    }
}

impl<A, NaD, NdD> Clone for GridMapN<A, NaD, NdD>
where
    A: na::Scalar,
    NaD: na::DimName,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    na::DefaultAllocator: na::allocator::Allocator<A, NaD>
        + na::allocator::Allocator<f32, NaD>
        + na::allocator::Allocator<isize, NaD>,
{
    fn clone(&self) -> Self {
        GridMapN {
            data: self.data.clone(),
            resolution: self.resolution,
            bounds: self.bounds.clone(),
        }
    }
}

fn ndarray_with_bounds<A, NaD, NdD>(
    resolution: f32,
    bounds: &Bounds<NaD>,
    default_value: A,
) -> nd::Array<A, NdD>
where
    A: na::Scalar,
    NaD: na::DimName,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    na::DefaultAllocator: na::allocator::Allocator<A, NaD>
        + na::allocator::Allocator<f32, NaD>
        + na::allocator::Allocator<isize, NaD>,
{
    let size_cells = geom::converter::cell_from_point(&bounds.max, &bounds.min, resolution);
    let vec_size: isize = size_cells.iter().product();
    let array_vec = std::vec::from_elem(default_value, vec_size as usize);
    return nd::Array::<A, NdD>::from_shape_vec(size_cells.to_ndindex(), array_vec).unwrap();
}

fn resized_ndarray2<A>(
    data: &nd::Array<A, nd::Ix2>,
    from_bounds: &Bounds<na::U2>,
    to_bounds: &Bounds<na::U2>,
    resolution: f32,
    default_value: A,
) -> nd::Array<A, nd::Ix2>
where
    A: na::Scalar,
{
    let mut to_data = ndarray_with_bounds(resolution, to_bounds, default_value);
    let overlapping_bounds = from_bounds.overlapping(to_bounds);
    if overlapping_bounds != Bounds::<na::U2>::empty() {
        let from_cell_min =
            geom::converter::cell_from_point(&overlapping_bounds.min, &from_bounds.min, resolution);
        let from_cell_max =
            geom::converter::cell_from_point(&overlapping_bounds.max, &from_bounds.min, resolution);
        let to_cell_min =
            geom::converter::cell_from_point(&overlapping_bounds.min, &to_bounds.min, resolution);
        let to_cell_max =
            geom::converter::cell_from_point(&overlapping_bounds.max, &to_bounds.min, resolution);
        let mut to_slice = to_data.slice_mut(nd::s![
            to_cell_min.coords[0]..to_cell_max.coords[0],
            to_cell_min.coords[1]..to_cell_max.coords[1]
        ]);
        let from_slice = data.slice(nd::s![
            from_cell_min.coords[0]..from_cell_max.coords[0],
            from_cell_min.coords[1]..from_cell_max.coords[1]
        ]);
        // TODO(kgreenek): This logic is susceptible to floating point errors that can cause the
        // sizes of slices to be different. Handle this intelligently.
        to_slice.assign(&from_slice);
    }
    to_data
}

fn resized_ndarray3<A>(
    data: &nd::Array<A, nd::Ix3>,
    from_bounds: &Bounds<na::U3>,
    to_bounds: &Bounds<na::U3>,
    resolution: f32,
    default_value: A,
) -> nd::Array<A, nd::Ix3>
where
    A: na::Scalar,
{
    let mut to_data = ndarray_with_bounds(resolution, to_bounds, default_value);
    let overlapping_bounds = from_bounds.overlapping(to_bounds);
    if overlapping_bounds != Bounds::<na::U3>::empty() {
        let from_cell_min =
            geom::converter::cell_from_point(&overlapping_bounds.min, &from_bounds.min, resolution);
        let from_cell_max =
            geom::converter::cell_from_point(&overlapping_bounds.max, &from_bounds.min, resolution);
        let to_cell_min =
            geom::converter::cell_from_point(&overlapping_bounds.min, &to_bounds.min, resolution);
        let to_cell_max =
            geom::converter::cell_from_point(&overlapping_bounds.max, &to_bounds.min, resolution);
        let mut to_slice = to_data.slice_mut(nd::s![
            to_cell_min.coords[0]..to_cell_max.coords[0],
            to_cell_min.coords[1]..to_cell_max.coords[1],
            to_cell_min.coords[2]..to_cell_max.coords[2]
        ]);
        let from_slice = data.slice(nd::s![
            from_cell_min.coords[0]..from_cell_max.coords[0],
            from_cell_min.coords[1]..from_cell_max.coords[1],
            from_cell_min.coords[2]..from_cell_max.coords[2]
        ]);
        // TODO(kgreenek): This logic is susceptible to floating point errors that can cause the
        // sizes of slices to be different. Handle this intelligently.
        to_slice.assign(&from_slice);
    }
    to_data
}

#[cfg(test)]
mod tests {
    use crate as kuba;
    use kuba::prelude::*;

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
            kuba::GridMap2f::from_bounds(0.1, kuba::bounds2![[0.0, 0.0], [0.2, 0.2]], 1.0);
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
            kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
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
            kuba::GridMap2f::from_bounds(0.1, kuba::bounds2![[0.0, 0.0], [0.2, 0.2]], 1.0);
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
            kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
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
            kuba::GridMap2f::from_bounds(0.1, kuba::bounds2![[0.0, 0.0], [0.2, 0.2]], 1.0);
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
            kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
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
