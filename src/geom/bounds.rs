use super::*;

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

#[derive(Clone, Debug, PartialEq)]
pub struct Bounds<NaD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD>,
{
    pub min: Point<NaD>,
    pub max: Point<NaD>,
}
pub type Bounds2 = Bounds<na::U2>;
pub type Bounds3 = Bounds<na::U3>;

impl<NaD> Bounds<NaD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<isize, NaD> + na::allocator::Allocator<f32, NaD>,
{
    pub fn new(min: Point<NaD>, max: Point<NaD>) -> Self {
        for (min_val, max_val) in min.iter().zip(max.iter()) {
            assert!(*min_val <= *max_val);
        }
        Bounds { min: min, max: max }
    }

    pub fn empty() -> Self {
        Bounds::new(Point::<NaD>::origin(), Point::<NaD>::origin())
    }

    pub fn discretized(&self, resolution: f32) -> Self {
        Bounds::new(
            Point::<NaD>::from(self.min.coords.map(|val| {
                let cell = val / resolution;
                if approx::relative_eq!(cell, cell.round()) {
                    return val;
                }
                cell.floor() * resolution
            })),
            Point::<NaD>::from(self.max.coords.map(|val| {
                let cell = val / resolution;
                if approx::relative_eq!(cell, cell.round()) {
                    return val;
                }
                cell.ceil() * resolution
            })),
        )
    }

    pub fn enclosing(&self, other: &Bounds<NaD>) -> Self {
        if *self == Bounds::empty() {
            return other.clone();
        }
        let mut bounds = other.clone();
        for (i, val) in self.min.coords.iter().enumerate() {
            if *val < other.min[i] {
                bounds.min[i] = *val;
            }
        }
        for (i, val) in self.max.coords.iter().enumerate() {
            if *val > other.max[i] {
                bounds.max[i] = *val;
            }
        }
        bounds
    }
}

impl<NaD> approx::AbsDiffEq for Bounds<NaD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<isize, NaD> + na::allocator::Allocator<f32, NaD>,
{
    type Epsilon = <f32 as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.min.abs_diff_eq(&other.min, epsilon) && self.max.abs_diff_eq(&other.max, epsilon)
    }
}

impl<NaD> approx::RelativeEq for Bounds<NaD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<isize, NaD> + na::allocator::Allocator<f32, NaD>,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.min.relative_eq(&other.min, epsilon, max_relative)
            && self.max.relative_eq(&other.max, epsilon, max_relative)
    }
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

    #[test]
    fn enclosing2_nominal() {
        let bounds1 = kuba::bounds2![[-0.02, 0.1], [1.1, 5.05]];
        let bounds2 = kuba::bounds2![[0.0, 0.2], [1.2, 6.05]];
        let expected_bounds = kuba::bounds2![[-0.02, 0.1], [1.2, 6.05]];
        assert_eq!(bounds1.enclosing(&bounds2), expected_bounds);
    }

    #[test]
    fn enclosing3_nominal() {
        let bounds1 = kuba::bounds3![[-0.02, 0.1, -1.31], [1.1, 5.05, 0.1]];
        let bounds2 = kuba::bounds3![[0.0, 0.2, 0.0], [1.2, 6.05, 1.0]];
        let expected_bounds = kuba::bounds3![[-0.02, 0.1, -1.31], [1.2, 6.05, 1.0]];
        assert_eq!(bounds1.enclosing(&bounds2), expected_bounds);
    }

    #[test]
    fn enclosing2_contained1() {
        let bounds1 = kuba::bounds2![[-0.02, 0.1], [1.1, 5.05]];
        let bounds2 = kuba::bounds2![[0.0, 0.2], [1.0, 4.05]];
        let expected_bounds = kuba::bounds2![[-0.02, 0.1], [1.1, 5.05]];
        assert_eq!(bounds1.enclosing(&bounds2), expected_bounds);
    }

    #[test]
    fn enclosing3_contained1() {
        let bounds1 = kuba::bounds3![[-0.02, 0.1, -1.31], [1.1, 5.05, 0.1]];
        let bounds2 = kuba::bounds3![[0.0, 0.2, 0.0], [1.0, 4.05, 0.05]];
        let expected_bounds = kuba::bounds3![[-0.02, 0.1, -1.31], [1.1, 5.05, 0.1]];
        assert_eq!(bounds1.enclosing(&bounds2), expected_bounds);
    }

    #[test]
    fn enclosing2_contained2() {
        let bounds1 = kuba::bounds2![[0.0, 0.2], [1.0, 4.05]];
        let bounds2 = kuba::bounds2![[-0.02, 0.1], [1.1, 5.05]];
        let expected_bounds = kuba::bounds2![[-0.02, 0.1], [1.1, 5.05]];
        assert_eq!(bounds1.enclosing(&bounds2), expected_bounds);
    }

    #[test]
    fn enclosing3_contained2() {
        let bounds1 = kuba::bounds3![[0.0, 0.2, 0.0], [1.0, 4.05, 0.05]];
        let bounds2 = kuba::bounds3![[-0.02, 0.1, -1.31], [1.1, 5.05, 0.1]];
        let expected_bounds = kuba::bounds3![[-0.02, 0.1, -1.31], [1.1, 5.05, 0.1]];
        assert_eq!(bounds1.enclosing(&bounds2), expected_bounds);
    }

    #[test]
    fn enclosing2_empty() {
        let bounds1 = kuba::Bounds2::empty();
        let bounds2 = kuba::bounds2![[-0.02, 0.1], [1.1, 5.05]];
        let expected_bounds = kuba::bounds2![[-0.02, 0.1], [1.1, 5.05]];
        assert_eq!(bounds1.enclosing(&bounds2), expected_bounds);
    }

    #[test]
    fn enclosing3_empty() {
        let bounds1 = kuba::Bounds3::empty();
        let bounds2 = kuba::bounds3![[-0.02, 0.1, -1.31], [1.1, 5.05, 0.1]];
        let expected_bounds = kuba::bounds3![[-0.02, 0.1, -1.31], [1.1, 5.05, 0.1]];
        assert_eq!(bounds1.enclosing(&bounds2), expected_bounds);
    }
}
