use crate::geom;
use crate::geom::Bounds;
use crate::geom::Bounds2;
use crate::geom::Bounds3;
use crate::geom::Cell;
use crate::geom::CellBounds2;
use crate::geom::CellBounds3;
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

pub const DEFAULT_RESOLUTION: f32 = 0.1;
pub const DEFAULT_TILE_SIZE: usize = 100;

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
    fn get(&self, cell: &Cell<NaD>) -> A;
    fn set(&mut self, cell: &Cell<NaD>, value: A);
    fn resolution(&self) -> f32;
    fn bounds(&self) -> Bounds<NaD>;

    /// Returns the point at the center of the given cell.
    /// We use the center of the cell instead of the top left corner to avoid issues with floating
    /// point rounding.
    fn point_from_cell(&self, cell: &Cell<NaD>) -> Point<NaD>;

    /// Returns the cell corresponding to the given point.
    /// If the point lies exactly on a cell boundary, the higher cell is returned.
    fn cell_from_point(&self, point: &Point<NaD>) -> Cell<NaD>;

    #[inline]
    fn track_changes(&self) -> bool;

    #[inline]
    fn set_track_changes(&mut self, value: bool);

    #[inline]
    fn changed_cells(&self) -> Vec<Cell<NaD>>;

    #[inline]
    fn clear_changed_cells(&mut self);

    #[inline]
    fn add_changed_cells(&mut self, cells: Vec<Cell<NaD>>);
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

// TODO(kgreenek): It's annoying to have to expose NaD and NdD. Figure out a way to just have one
// generic dimension parameter.
pub struct GridMapN<A, NaD, NdD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
{
    pub data: nd::Array<A, NdD>,
    pub resolution: f32,
    pub bounds: Bounds<NaD>,
    pub tile_size: usize,

    track_changes: bool,
    changed_cells: Vec<Cell<NaD>>,
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
    pub fn new(resolution: f32, bounds: Bounds<NaD>, tile_size: usize, default_value: A) -> Self {
        GridMapN {
            data: ndarray_with_bounds(resolution, &bounds, default_value),
            resolution: resolution,
            bounds: bounds,
            tile_size: tile_size,
            track_changes: false,
            changed_cells: vec![],
        }
    }

    pub fn default() -> Self {
        GridMapN {
            data: nd::Array::<A, NdD>::from_shape_vec(NdD::zeros(NdD::NDIM.unwrap()), vec![])
                .unwrap(),
            resolution: DEFAULT_RESOLUTION,
            bounds: Bounds::empty(),
            tile_size: DEFAULT_TILE_SIZE,
            track_changes: false,
            changed_cells: vec![],
        }
    }

    /// You almost certainly don't want this. This is for tests.
    pub fn from_ndarray(
        ndarray: nd::Array<A, NdD>,
        resolution: f32,
        bounds: Bounds<NaD>,
        tile_size: usize,
    ) -> Self {
        GridMapN {
            data: ndarray,
            resolution: resolution,
            bounds: bounds,
            tile_size: tile_size,
            track_changes: false,
            changed_cells: vec![],
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

    #[inline]
    fn track_changes(&self) -> bool {
        self.track_changes
    }

    #[inline]
    fn set_track_changes(&mut self, value: bool) {
        self.track_changes = value;
    }

    #[inline]
    fn changed_cells(&self) -> Vec<Cell<NaD>> {
        self.changed_cells.clone()
    }

    #[inline]
    fn clear_changed_cells(&mut self) {
        self.changed_cells.clear();
    }

    #[inline]
    fn add_changed_cells(&mut self, cells: Vec<Cell<NaD>>) {
        self.changed_cells.extend(cells);
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
        let new_cell_bounds =
            CellBounds2::from_bounds(&new_bounds, self.resolution).discretized(self.tile_size);
        let new_bounds = Bounds2::from_cell_bounds(&new_cell_bounds, self.resolution);
        let cell_bounds = CellBounds2::from_bounds(&self.bounds, self.resolution);
        if new_cell_bounds == cell_bounds {
            return;
        }
        // This is super helpful for debugging.
        //println!("Resizing GridMap ---------------------------------------------------");
        //println!("tile_size: {}", self.tile_size);
        //println!("Bounds:\nfrom: {:?}\nto  : {:?}", self.bounds, new_bounds);
        //println!(
        //    "Cells :\nfrom: {:?}\nto  : {:?}",
        //    cell_bounds, new_cell_bounds
        //);
        let mut to_data = ndarray_with_bounds(self.resolution, &new_bounds, default_value);
        if self.bounds != Bounds2::empty() {
            let offset = cell_bounds.min - new_cell_bounds.min;
            let cells_size = cell_bounds.max - cell_bounds.min;
            assert!(offset.x >= 0 && offset.y >= 0);
            let mut to_slice = to_data.slice_mut(nd::s![
                offset.x..(cells_size.x + offset.x),
                offset.y..(cells_size.y + offset.y),
            ]);
            let from_slice = self.data.slice(nd::s![0..cells_size.x, 0..cells_size.y,]);
            to_slice.assign(&from_slice);
        }
        self.bounds = new_bounds;
        self.data = to_data;
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
        let new_cell_bounds =
            CellBounds3::from_bounds(&new_bounds, self.resolution).discretized(self.tile_size);
        let new_bounds = Bounds3::from_cell_bounds(&new_cell_bounds, self.resolution);
        let cell_bounds = CellBounds3::from_bounds(&self.bounds, self.resolution);
        if new_cell_bounds == cell_bounds {
            return;
        }
        // This is super helpful for debugging.
        //println!("Resizing GridMap ---------------------------------------------------");
        //println!("tile_size: {}", self.tile_size);
        //println!("Bounds:\nfrom: {:?}\nto  : {:?}", self.bounds, new_bounds);
        //println!(
        //    "Cells :\nfrom: {:?}\nto  : {:?}",
        //    cell_bounds, new_cell_bounds
        //);
        let mut to_data = ndarray_with_bounds(self.resolution, &new_bounds, default_value);
        if self.bounds != Bounds3::empty() {
            let offset = cell_bounds.min - new_cell_bounds.min;
            let cells_size = cell_bounds.max - cell_bounds.min;
            assert!(offset.x >= 0 && offset.y >= 0);
            let mut to_slice = to_data.slice_mut(nd::s![
                offset.x..(cells_size.x + offset.x),
                offset.y..(cells_size.y + offset.y),
                offset.z..(cells_size.z + offset.z),
            ]);
            let from_slice =
                self.data
                    .slice(nd::s![0..cells_size.x, 0..cells_size.y, 0..cells_size.z,]);
            to_slice.assign(&from_slice);
        }
        self.bounds = new_bounds;
        self.data = to_data;
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
            tile_size: self.tile_size,
            track_changes: self.track_changes,
            changed_cells: self.changed_cells.clone(),
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

// This is unused due to the floating point errors.
fn _resized_ndarray2<A>(
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

// This is unused due to the floating point errors.
fn _resized_ndarray3<A>(
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

    const TILE_SIZE: usize = 2;

    #[test]
    fn get2() {
        let grid_map = kuba::GridMap2f::from_ndarray(
            nd::array![[1.0, 2.0], [10.0, 20.0]],
            0.1,
            kuba::bounds2![[0.0, 0.0], [0.2, 0.2]],
            TILE_SIZE,
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
            TILE_SIZE,
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
            TILE_SIZE,
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
            TILE_SIZE,
        );
        grid_map.set(&kuba::cell3![0, 0, 0], 15.0);
        grid_map.set(&kuba::cell3![0, 1, 0], 25.0);
        assert_eq!(grid_map.get(&kuba::cell3![0, 0, 0]), 15.0);
        assert_eq!(grid_map.get(&kuba::cell3![0, 1, 0]), 25.0);
    }

    #[test]
    fn expand_bounds2_nominal() {
        let mut grid_map =
            kuba::GridMap2f::new(0.1, kuba::bounds2![[0.0, 0.0], [0.2, 0.2]], TILE_SIZE, 1.0);
        grid_map.expand_bounds(&kuba::bounds2![[0.1, 0.1], [0.3, 0.3]], 0.0);
        let mut expected_data = nd::Array2::<f32>::zeros((4, 4));
        expected_data[(0, 0)] = 1.0;
        expected_data[(0, 1)] = 1.0;
        expected_data[(1, 0)] = 1.0;
        expected_data[(1, 1)] = 1.0;
        assert!(approx::relative_eq!(
            grid_map.bounds(),
            kuba::bounds2![[0.0, 0.0], [0.4, 0.4]]
        ));
        assert_eq!(grid_map.data, expected_data);

        let mut grid_map =
            kuba::GridMap2f::new(0.1, kuba::bounds2![[0.0, 0.0], [0.2, 0.2]], TILE_SIZE, 1.0);
        grid_map.expand_bounds(&kuba::bounds2![[-0.1, -0.1], [0.1, 0.1]], 0.0);
        let mut expected_data = nd::Array2::<f32>::zeros((4, 4));
        expected_data[(2, 2)] = 1.0;
        expected_data[(2, 3)] = 1.0;
        expected_data[(3, 2)] = 1.0;
        expected_data[(3, 3)] = 1.0;
        assert!(approx::relative_eq!(
            grid_map.bounds(),
            kuba::bounds2![[-0.2, -0.2], [0.2, 0.2]]
        ));
        assert_eq!(grid_map.data, expected_data);
    }

    #[test]
    fn expand_bounds3_nominal() {
        let mut grid_map = kuba::GridMap3f::new(
            0.1,
            kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
            TILE_SIZE,
            1.0,
        );
        grid_map.expand_bounds(&kuba::bounds3![[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]], 0.0);
        let mut expected_data = nd::Array3::<f32>::zeros((4, 4, 4));
        expected_data[(0, 0, 0)] = 1.0;
        expected_data[(0, 0, 1)] = 1.0;
        expected_data[(0, 1, 0)] = 1.0;
        expected_data[(0, 1, 1)] = 1.0;
        expected_data[(1, 0, 0)] = 1.0;
        expected_data[(1, 0, 1)] = 1.0;
        expected_data[(1, 1, 0)] = 1.0;
        expected_data[(1, 1, 1)] = 1.0;
        assert!(approx::relative_eq!(
            grid_map.bounds(),
            kuba::bounds3![[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]]
        ));
        assert_eq!(grid_map.data, expected_data);

        let mut grid_map = kuba::GridMap3f::new(
            0.1,
            kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
            TILE_SIZE,
            1.0,
        );
        grid_map.expand_bounds(&kuba::bounds3![[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]], 0.0);
        let mut expected_data = nd::Array3::<f32>::zeros((4, 4, 4));
        expected_data[(2, 2, 2)] = 1.0;
        expected_data[(2, 2, 3)] = 1.0;
        expected_data[(2, 3, 2)] = 1.0;
        expected_data[(2, 3, 3)] = 1.0;
        expected_data[(3, 2, 2)] = 1.0;
        expected_data[(3, 2, 3)] = 1.0;
        expected_data[(3, 3, 2)] = 1.0;
        expected_data[(3, 3, 3)] = 1.0;
        assert!(approx::relative_eq!(
            grid_map.bounds(),
            kuba::bounds3![[-0.2, -0.2, -0.2], [0.2, 0.2, 0.2]]
        ));
        assert_eq!(grid_map.data, expected_data);
    }

    #[test]
    fn expand_bounds2_empty() {
        let mut grid_map =
            kuba::GridMap2f::new(0.1, kuba::bounds2![[0.0, 0.0], [0.0, 0.0]], TILE_SIZE, 1.0);
        grid_map.expand_bounds(&kuba::bounds2![[0.3, 0.3], [0.4, 0.4]], 0.0);
        assert!(approx::relative_eq!(
            grid_map.bounds(),
            kuba::bounds2![[0.2, 0.2], [0.6, 0.6]]
        ));
        assert_eq!(grid_map.data, nd::Array2::<f32>::zeros((4, 4)));
    }

    #[test]
    fn expand_bounds3_empty() {
        let mut grid_map = kuba::GridMap3f::new(
            0.1,
            kuba::bounds3![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            TILE_SIZE,
            1.0,
        );
        grid_map.expand_bounds(&kuba::bounds3![[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]], 0.0);
        assert!(approx::relative_eq!(
            grid_map.bounds(),
            kuba::bounds3![[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]]
        ));
        assert_eq!(grid_map.data, nd::Array3::<f32>::zeros((4, 4, 4)));
    }
}
