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
