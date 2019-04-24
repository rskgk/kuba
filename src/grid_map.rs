use crate::geom::Bounds;
use crate::geom::Cell;
use crate::geom::CellToNdIndex;
//use crate::geom::Point;

pub type GridMap2<A> = GridMapN<A, na::U2, nd::Ix2>;
pub type GridMap2f = GridMap2<f32>;
pub type GridMap2i = GridMap2<i32>;
pub type GridMap2b = GridMap2<bool>;

pub type GridMap3<A> = GridMapN<A, na::U3, nd::Ix3>;
pub type GridMap3f = GridMap3<f32>;
pub type GridMap3i = GridMap2<i32>;
pub type GridMap3b = GridMap2<bool>;

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
        + na::allocator::Allocator<usize, NaD>
        + na::allocator::Allocator<f32, NaD>,
{
    pub fn from_ndarray(ndarray: nd::Array<A, NdD>, resolution: f32, bounds: Bounds<NaD>) -> Self {
        GridMapN {
            data: ndarray,
            resolution: resolution,
            bounds: bounds,
        }
    }

    //pub fn point_from_cell(&self, cell: &Cell<NaD>) -> Point<NaD> {
    //}

    /// Returns the value at the given cell.
    #[inline]
    pub fn get(&self, cell: Cell<NaD>) -> A {
        self.data[cell.to_ndindex()]
    }

    /// Sets the value at the given cell.
    #[inline]
    pub fn set(&mut self, cell: Cell<NaD>, value: A) {
        self.data[cell.to_ndindex()] = value;
    }
}

#[cfg(test)]
mod tests {
    use crate as kuba;

    #[test]
    fn get() {
        let grid_map = kuba::GridMap2f::from_ndarray(
            nd::array![[1.0, 2.0], [10.0, 20.0]],
            0.1,
            kuba::Bounds2 {
                min: kuba::Point2::new(0.0, 0.0),
                max: kuba::Point2::new(0.2, 0.2),
            },
        );

        assert_eq!(grid_map.get(kuba::Cell2::new(0, 0)), 1.0);
        assert_eq!(grid_map.get(kuba::Cell2::new(0, 1)), 2.0);
        assert_eq!(grid_map.get(kuba::Cell2::new(1, 0)), 10.0);
        assert_eq!(grid_map.get(kuba::Cell2::new(1, 1)), 20.0);
    }

    #[test]
    fn set() {
        let mut grid_map = kuba::GridMap2f::from_ndarray(
            nd::array![[1.0, 2.0], [10.0, 20.0]],
            0.1,
            kuba::Bounds2 {
                min: kuba::Point2::new(0.0, 0.0),
                max: kuba::Point2::new(0.2, 0.2),
            },
        );

        grid_map.set(kuba::Cell2::new(0, 0), 15.0);
        assert_eq!(grid_map.get(kuba::Cell2::new(0, 0)), 15.0);
    }
}
