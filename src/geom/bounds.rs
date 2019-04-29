use super::*;

#[derive(Debug, PartialEq)]
pub struct Bounds<D>
where
    D: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<f32, D>,
{
    pub min: Point<D>,
    pub max: Point<D>,
}
pub type Bounds2 = Bounds<na::U2>;
pub type Bounds3 = Bounds<na::U3>;

impl<D> Bounds<D>
where
    D: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<f32, D>,
{
    pub fn new(min: Point<D>, max: Point<D>) -> Self {
        for (min_val, max_val) in min.iter().zip(max.iter()) {
            assert!(*min_val < *max_val);
        }
        Bounds{min: min, max: max}
    }

    pub fn discretized(&self, resolution: f32) -> Self {
        Bounds::new(
            Point::<D>::from(self.min.coords.map(|val| {
                let cell = val / resolution;
                if approx::relative_eq!(cell, cell.round()) {
                    return val
                }
                cell.floor() * resolution
            })),
            Point::<D>::from(self.max.coords.map(|val| {
                let cell = val / resolution;
                if approx::relative_eq!(cell, cell.round()) {
                    return val
                }
                cell.ceil() * resolution
            }))
        )
    }
}

#[macro_export]
macro_rules! bounds2 {
    ([$($min: expr),+], [$($max: expr),+]) => {{
        $crate::Bounds2{min: $crate::Point2::new($($min),*), max: $crate::Point2::new($($max),*)}
    }}
}

#[macro_export]
macro_rules! bounds3 {
    ([$($min: expr),+], [$($max: expr),+]) => {{
        $crate::Bounds3{min: $crate::Point3::new($($min),*), max: $crate::Point3::new($($max),*)}
    }}
}

#[cfg(test)]
mod tests {
    use crate as kuba;

    #[test]
    fn discretized2() {
        let bounds = kuba::bounds2![[-0.02, 0.1], [1.1, 5.05]];
        let expected_bounds = kuba::bounds2![[-0.1, 0.1], [1.1, 5.1]];
        assert_eq!(bounds.discretized(0.1), expected_bounds);
    }

    #[test]
    fn discretized3() {
        let bounds = kuba::bounds3![[-0.02, 0.1, -1.31], [1.1, 5.05, -1.3]];
        let expected_bounds = kuba::bounds3![[-0.1, 0.1, -1.4], [1.1, 5.1, -1.3]];
        assert_eq!(bounds.discretized(0.1), expected_bounds);
    }
}
