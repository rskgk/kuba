use crate::geom::Bounds;
use crate::geom::Cell;
use crate::geom::CellToNdIndex;
use crate::geom::Point;
use crate::geom::PointCloud;
use crate::map::grid_map::GridMap;
use crate::map::GridMapN;

// TODO(kgreenek): It's annoying to have to expose NaD and NdD. Figure out a way to just have one
// generic dimention parameter.
pub struct OccupancyGridMapN<NaD, NdD>
where
    NaD: na::DimName,
    NdD: nd::Dimension,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
{
    grid_map: GridMapN<f32, NaD, NdD>,
}

impl<NaD, NdD> OccupancyGridMapN<NaD, NdD>
where
    NaD: na::DimName + std::hash::Hash,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    <na::DefaultAllocator as na::allocator::Allocator<isize, NaD>>::Buffer: std::hash::Hash,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
{
    pub fn from_ndarray(
        ndarray: nd::Array<f32, NdD>,
        resolution: f32,
        bounds: Bounds<NaD>,
    ) -> Self {
        OccupancyGridMapN {
            grid_map: GridMapN::from_ndarray(ndarray, resolution, bounds),
        }
    }

    pub fn from_bounds(resolution: f32, bounds: &Bounds<NaD>, default_value: f32) -> Self {
        OccupancyGridMapN {
            grid_map: GridMapN::from_bounds(resolution, bounds, default_value),
        }
    }

    pub fn integrate_point_cloud(&mut self, origin: &Point<NaD>, point_cloud: &PointCloud<NaD>) {}
}

impl<NaD, NdD> GridMap<f32, NaD, NdD> for OccupancyGridMapN<NaD, NdD>
where
    NaD: na::DimName + std::hash::Hash,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    <na::DefaultAllocator as na::allocator::Allocator<isize, NaD>>::Buffer: std::hash::Hash,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
{
    #[inline]
    fn get(&self, cell: &Cell<NaD>) -> f32 {
        self.grid_map.get(cell)
    }

    #[inline]
    fn set(&mut self, cell: &Cell<NaD>, value: f32) {
        self.grid_map.set(cell, value);
    }

    #[inline]
    fn resolution(&self) -> f32 {
        self.grid_map.resolution
    }

    #[inline]
    fn bounds(&self) -> Bounds<NaD> {
        self.grid_map.bounds.clone()
    }

    #[inline]
    fn point_from_cell(&self, cell: &Cell<NaD>) -> Point<NaD> {
        self.grid_map.point_from_cell(cell)
    }

    #[inline]
    fn cell_from_point(&self, point: &Point<NaD>) -> Cell<NaD> {
        self.grid_map.cell_from_point(point)
    }
}
