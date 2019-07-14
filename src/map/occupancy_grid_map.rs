use crate::geom::Bounds;
use crate::geom::Cell;
use crate::geom::CellToNdIndex;
use crate::geom::Point;
use crate::geom::PointCloud;
use crate::map::grid_map;
use crate::map::grid_map::ExpandableGridMap;
use crate::map::grid_map::GridMap;
use crate::map::GridMapN;
use crate::map::LidarNoiseModel;
use crate::map::NoiseModel;

pub type LidarOccupancyGridMap<NaD, NdD> = OccupancyGridMapN<LidarNoiseModel, NaD, NdD>;
pub type LidarOccupancyGridMap2 = LidarOccupancyGridMap<na::U2, nd::Ix2>;
pub type LidarOccupancyGridMap3 = LidarOccupancyGridMap<na::U3, nd::Ix3>;

// TODO(kgreenek): It's annoying to have to expose NaD and NdD. Figure out a way to just have one
// generic dimention parameter.
pub struct OccupancyGridMapN<NM, NaD, NdD>
where
    NM: NoiseModel<NaD, NdD>,
    NaD: na::DimName,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
{
    pub grid_map: GridMapN<f32, NaD, NdD>,
    pub noise_model: NM,
}

impl<NM, NaD, NdD> OccupancyGridMapN<NM, NaD, NdD>
where
    NM: NoiseModel<NaD, NdD>,
    NaD: na::DimName + std::hash::Hash,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    GridMapN<f32, NaD, NdD>: ExpandableGridMap<f32, NaD, NdD>,
    <na::DefaultAllocator as na::allocator::Allocator<isize, NaD>>::Buffer: std::hash::Hash,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
{
    pub fn default() -> Self {
        Self::new(
            NM::default_noise_model(),
            grid_map::DEFAULT_RESOLUTION,
            grid_map::DEFAULT_TILE_SIZE,
        )
    }

    pub fn new(noise_model: NM, resolution: f32, tile_size: usize) -> Self {
        OccupancyGridMapN {
            grid_map: GridMapN::new(
                resolution,
                Bounds::empty(),
                tile_size,
                noise_model.default_cell_value(),
            ),
            noise_model: noise_model,
        }
    }

    pub fn integrate_point_cloud(&mut self, origin: &Point<NaD>, point_cloud: &PointCloud<NaD>) {
        self.noise_model
            .integrate_point_cloud(&mut self.grid_map, origin, point_cloud);
    }

    pub fn occupied(&self, cell: &Cell<NaD>) -> bool {
        self.noise_model.occupied(&self.grid_map, cell)
    }
}

impl<NM, NaD, NdD> GridMap<f32, NaD, NdD> for OccupancyGridMapN<NM, NaD, NdD>
where
    NM: NoiseModel<NaD, NdD>,
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
