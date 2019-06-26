use crate::geom::Cell;
use crate::geom::CellToNdIndex;
use crate::geom::Point;
use crate::geom::PointCloud;
use crate::map::GridMap;

pub trait NoiseModel<NaD, NdD>
where
    NaD: na::DimName,
    NdD: nd::Dimension,
    Cell<NaD>: CellToNdIndex<NaD, NdD>,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
{
    /// Returns the default value for a cell whose probability is unknown.
    #[inline]
    fn default_cell_value(&self) -> f32;

    /// Updates the probability values in the given grid map from the given point cloud according
    /// to the properties of the noise model.
    fn integrate_point_cloud(
        &self,
        grid_map: &mut impl GridMap<f32, NaD, NdD>,
        origin: &Point<NaD>,
        point_cloud: &PointCloud<NaD>,
    );
}
