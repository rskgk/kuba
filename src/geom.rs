pub type Vector<D> = na::VectorN<f32, D>;
pub type Vector2 = Vector<na::U2>;
pub type Vector3 = Vector<na::U3>;

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

pub type Point<D> = na::Point<f32, D>;
pub type Point2 = Point<na::U2>;
pub type Point3 = Point<na::U3>;

#[macro_export]
macro_rules! point2 {
    ($($val: expr),+) => {{
        $crate::Point2::new($($val),*)
    }}
}

#[macro_export]
macro_rules! point3 {
    ($($val: expr),+) => {{
        $crate::Point3::new($($val),*)
    }}
}

pub type Cell<D> = na::Point<usize, D>;
pub type Cell2 = Cell<na::U2>;
pub type Cell3 = Cell<na::U3>;

pub trait CellToNdIndex<NaD, NdD>
where
    NaD: na::DimName,
    NdD: nd::Dimension,
    na::DefaultAllocator: na::allocator::Allocator<usize, NaD>,
{
    #[inline]
    fn to_ndindex(&self) -> NdD;
}

#[macro_export]
macro_rules! cell2 {
    ($($val: expr),+) => {{
        $crate::Cell2::new($($val),*)
    }}
}

#[macro_export]
macro_rules! cell3 {
    ($($val: expr),+) => {{
        $crate::Cell3::new($($val),*)
    }}
}

macro_rules! ndindex_from_cell_impl {
    ($($NaD: ty, $NdD: ty, $len: expr);*) => {$(
        impl CellToNdIndex<$NaD, $NdD> for Cell<$NaD> {
            #[inline]
            fn to_ndindex(&self) -> $NdD {
                let index: [usize; $len] = self.coords.into();
                nd::Dim(index)
            }
        }
    )*}
}

ndindex_from_cell_impl!(na::U2, nd::Ix2, 2; na::U3, nd::Ix3, 3);

