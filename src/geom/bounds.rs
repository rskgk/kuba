use super::*;

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
