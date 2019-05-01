pub type Vector<NaD> = na::VectorN<f32, NaD>;
pub type Vector2 = Vector<na::U2>;
pub type Vector3 = Vector<na::U3>;

use crate::geom::math;

pub fn signum<NaD>(vec: &Vector<NaD>) -> Vector<NaD>
where
    NaD: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<f32, NaD> {
    Vector::<NaD>::from(vec.column(0).map(|coord| math::f32_signum(coord)))
}
