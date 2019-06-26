use crate::geom::Cell;
use crate::geom::CellToNdIndex;
use crate::geom::Point;
use crate::geom::PointCloud;
use crate::map::GridMap;
use crate::map::NoiseModel;
use crate::math;
use crate::ray_caster;

pub const DEFAULT_HIT_PROBABILITY: f32 = 0.7; // 0.85 in logodds
pub const DEFAULT_MISS_PROBABILITY: f32 = 0.4; // -0.4 in logodds
pub const DEFAULT_MIN_PROBABILITY: f32 = 0.1192; // -2.0 in logodds
pub const DEFAULT_MAX_PROBABILITY: f32 = 0.971; // 3.5 in logodds
pub const DEFAULT_OCCUPIED_THRESHOLD: f32 = 0.5; // 0.0 in logodds
pub const DEFAULT_MAX_RANGE: f32 = 10.0; // meters

/// An approximate noise model for lidar point clouds.
/// Values are stored in logodds to prevent semi-expensive log calculations for each cell update.
pub struct LidarNoiseModel {
    pub hit_probability_logodds: f32,
    pub miss_probability_logodds: f32,
    pub min_probability_logodds: f32,
    pub max_probability_logodds: f32,
    pub occupied_threshold_logodds: f32,
    pub max_range: f32,
}

impl LidarNoiseModel {
    pub fn new(
        hit_probability: f32,
        miss_probability: f32,
        min_probability: f32,
        max_probability: f32,
        occupied_threshold: f32,
        max_range: f32,
    ) -> Self {
        assert!(hit_probability >= 0.0 and hit_probability <= 1.0);
        assert!(miss_probability >= 0.0 and miss_probability <= 1.0);
        assert!(min_probability >= 0.0 and min_probability <= 1.0);
        assert!(max_probability >= 0.0 and max_probability <= 1.0);
        assert!(occupied_threshold >= 0.0 and occupied_threshold <= 1.0);
        LidarNoiseModel {
            hit_probability_logodds: math::logodds_from_probability(hit_probability),
            miss_probability_logodds: math::logodds_from_probability(miss_probability),
            min_probability_logodds: math::logodds_from_probability(min_probability),
            max_probability_logodds: math::logodds_from_probability(max_probability),
            occupied_threshold_logodds: math::logodds_from_probability(occupied_threshold),
            max_range: max_range,
        }
    }

    pub fn default() -> Self {
        LidarNoiseModel::new(
            DEFAULT_HIT_PROBABILITY,
            DEFAULT_MISS_PROBABILITY,
            DEFAULT_MIN_PROBABILITY,
            DEFAULT_MAX_PROBABILITY,
            DEFAULT_OCCUPIED_THRESHOLD,
            DEFAULT_MAX_RANGE,
        )
    }
}

impl LidarNoiseModel {
    pub fn integrate_hit<NaD, NdD>(
        &self,
        grid_map: &mut impl GridMap<f32, NaD, NdD>,
        cell: &Cell<NaD>,
    ) where
        NaD: na::DimName,
        NdD: nd::Dimension,
        Cell<NaD>: CellToNdIndex<NaD, NdD>,
        na::DefaultAllocator:
            na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
    {
        grid_map.set(
            &cell,
            (grid_map.get(&cell) + self.hit_probability_logodds).min(self.max_probability_logodds),
        );
    }

    pub fn integrate_miss<NaD, NdD>(
        &self,
        grid_map: &mut impl GridMap<f32, NaD, NdD>,
        cell: &Cell<NaD>,
    ) where
        NaD: na::DimName,
        NdD: nd::Dimension,
        Cell<NaD>: CellToNdIndex<NaD, NdD>,
        na::DefaultAllocator:
            na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
    {
        grid_map.set(
            &cell,
            (grid_map.get(&cell) + self.miss_probability_logodds).max(self.min_probability_logodds),
        );
    }
}

impl<NaD, NdD> NoiseModel<NaD, NdD> for LidarNoiseModel
where
    NaD: na::DimName + std::hash::Hash,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    <na::DefaultAllocator as na::allocator::Allocator<isize, NaD>>::Buffer: std::hash::Hash,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
{
    #[inline]
    fn default_cell_value(&self) -> f32 {
        0.0
    }

    fn integrate_point_cloud(
        &self,
        grid_map: &mut impl GridMap<f32, NaD, NdD>,
        origin: &Point<NaD>,
        point_cloud: &PointCloud<NaD>,
    ) {
        let (free_cells, occupied_cells) = ray_caster::cast_rays(
            origin,
            point_cloud,
            &grid_map.bounds(),
            grid_map.resolution(),
            self.max_range,
        );
        for cell in &free_cells {
            self.integrate_miss(grid_map, cell);
        }
        for cell in &occupied_cells {
            self.integrate_hit(grid_map, cell);
        }
    }
}
