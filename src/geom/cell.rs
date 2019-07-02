// NOTE: We make cell an isize instead of usize so that it can be manipulated with negative values,
// e.g. when using it as an offset instead of an absolute cell in a map.
pub type Cell<D> = na::Point<isize, D>;
pub type Cell2 = Cell<na::U2>;
pub type Cell3 = Cell<na::U3>;

pub trait CellToNdIndex<NaD, NdD>
where
    NaD: na::DimName,
    NdD: nd::Dimension,
    na::DefaultAllocator: na::allocator::Allocator<isize, NaD>,
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
                unsafe {
                    let index: [isize; $len] = self.coords.into();
                    // Consider removing this check for performance.
                    for val in &index[..] {
                        assert!(!val.is_negative(), "Negative value in cell {:?}", index);
                    }
                    nd::Dim(std::mem::transmute::<[isize; $len], [usize; $len]>(index))
                }
            }
        }
    )*}
}

ndindex_from_cell_impl!(na::U2, nd::Ix2, 2; na::U3, nd::Ix3, 3);
