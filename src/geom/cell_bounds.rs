use crate::geom::converter;
use crate::geom::Bounds;
use crate::geom::Cell;
use crate::geom::Point;

#[macro_export]
macro_rules! cell_bounds2 {
    ([$($min: expr),+], [$($max: expr),+]) => {{
        $crate::CellBounds2{min: $crate::Cell2::new($($min),*), max: $crate::Cell2::new($($max),*)}
    }}
}

#[macro_export]
macro_rules! cell_bounds3 {
    ([$($min: expr),+], [$($max: expr),+]) => {{
        $crate::CellBounds3{min: $crate::Cell3::new($($min),*), max: $crate::Cell3::new($($max),*)}
    }}
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellBounds<NaD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<isize, NaD>,
{
    pub min: Cell<NaD>,
    pub max: Cell<NaD>,
}
pub type CellBounds2 = CellBounds<na::U2>;
pub type CellBounds3 = CellBounds<na::U3>;

impl<NaD> CellBounds<NaD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> + na::allocator::Allocator<isize, NaD>,
{
    pub fn new(min: Cell<NaD>, max: Cell<NaD>) -> Self {
        for (min_val, max_val) in min.iter().zip(max.iter()) {
            assert!(*min_val <= *max_val);
        }
        CellBounds { min: min, max: max }
    }

    pub fn empty() -> Self {
        CellBounds::new(Cell::<NaD>::origin(), Cell::<NaD>::origin())
    }

    pub fn from_bounds(bounds: &Bounds<NaD>, resolution: f32) -> Self {
        CellBounds {
            min: converter::cell_from_point(&bounds.min, &Point::<NaD>::origin(), resolution),
            max: converter::cell_from_point(&bounds.max, &Point::<NaD>::origin(), resolution),
        }
    }

    /// Returns CellBounds that is rounded to the given tile_size.
    /// The returned CellBounds will be guaranteed to contain the original bounds.
    pub fn discretized(&self, tile_size: usize) -> Self {
        CellBounds {
            min: Cell::<NaD>::from(
                self.min
                    .coords
                    .map(|coord| coord - (coord % (tile_size as isize)).abs()),
            ),
            max: Cell::<NaD>::from(
                self.max
                    .coords
                    .map(|coord| coord + (coord % (tile_size as isize)).abs()),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate as kuba;

    #[test]
    fn from_bounds2() {
        let resolution = 0.1;
        let bounds = kuba::bounds2![[0.0, 0.0], [0.2, 0.2]];
        let cell_bounds = kuba::cell_bounds2![[0, 0], [2, 2]];
        assert_eq!(
            kuba::CellBounds2::from_bounds(&bounds, resolution),
            cell_bounds
        );
        let bounds = kuba::bounds2![[-0.3, -0.3], [-0.1, -0.1]];
        let cell_bounds = kuba::cell_bounds2![[-3, -3], [-1, -1]];
        assert_eq!(
            kuba::CellBounds2::from_bounds(&bounds, resolution),
            cell_bounds
        );
    }

    #[test]
    fn from_bounds3() {
        let resolution = 0.1;
        let bounds = kuba::bounds3![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]];
        let cell_bounds = kuba::cell_bounds3![[0, 0, 0], [2, 2, 2]];
        assert_eq!(
            kuba::CellBounds3::from_bounds(&bounds, resolution),
            cell_bounds
        );
        let bounds = kuba::bounds3![[-0.3, -0.3, -0.3], [-0.1, -0.1, -0.1]];
        let cell_bounds = kuba::cell_bounds3![[-3, -3, -3], [-1, -1, -1]];
        assert_eq!(
            kuba::CellBounds3::from_bounds(&bounds, resolution),
            cell_bounds
        );
    }

    #[test]
    fn discretized2() {
        let tile_size = 2;
        let cell_bounds = kuba::cell_bounds2![[0, 0], [2, 2]];
        assert_eq!(cell_bounds.discretized(tile_size), cell_bounds);
        let cell_bounds = kuba::cell_bounds2![[0, 0], [3, 3]];
        let expected_cell_bounds = kuba::cell_bounds2![[0, 0], [4, 4]];
        assert_eq!(cell_bounds.discretized(tile_size), expected_cell_bounds);
        let cell_bounds = kuba::cell_bounds2![[1, 1], [3, 3]];
        assert_eq!(cell_bounds.discretized(tile_size), expected_cell_bounds);
        let cell_bounds = kuba::cell_bounds2![[-1, -1], [3, 3]];
        let expected_cell_bounds = kuba::cell_bounds2![[-2, -2], [4, 4]];
        assert_eq!(cell_bounds.discretized(tile_size), expected_cell_bounds);
        let cell_bounds = kuba::cell_bounds2![[-3, -3], [-1, -1]];
        let expected_cell_bounds = kuba::cell_bounds2![[-4, -4], [0, 0]];
        assert_eq!(cell_bounds.discretized(tile_size), expected_cell_bounds);
    }

    #[test]
    fn discretized3() {
        let tile_size = 2;
        let cell_bounds = kuba::cell_bounds3![[0, 0, 0], [2, 2, 2]];
        assert_eq!(cell_bounds.discretized(tile_size), cell_bounds);
        let cell_bounds = kuba::cell_bounds3![[0, 0, 0], [3, 3, 3]];
        let expected_cell_bounds = kuba::cell_bounds3![[0, 0, 0], [4, 4, 4]];
        assert_eq!(cell_bounds.discretized(tile_size), expected_cell_bounds);
        let cell_bounds = kuba::cell_bounds3![[1, 1, 1], [3, 3, 3]];
        assert_eq!(cell_bounds.discretized(tile_size), expected_cell_bounds);
        let cell_bounds = kuba::cell_bounds3![[-1, -1, -1], [3, 3, 3]];
        let expected_cell_bounds = kuba::cell_bounds3![[-2, -2, -2], [4, 4, 4]];
        assert_eq!(cell_bounds.discretized(tile_size), expected_cell_bounds);
        let cell_bounds = kuba::cell_bounds3![[-3, -3, -3], [-1, -1, -1]];
        let expected_cell_bounds = kuba::cell_bounds3![[-4, -4, -4], [0, 0, 0]];
        assert_eq!(cell_bounds.discretized(tile_size), expected_cell_bounds);
    }
}
