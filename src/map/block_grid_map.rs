use std::hash::Hash;

use crate::geom::Cell;
use crate::geom::Cell3;
use crate::geom::CellToNdIndex;

pub struct Block3<A> {
    data: nd::Array3<A>,
}

pub struct OccupancyBlockGridMap3<A> {
    blocks: std::collections::HashMap<i32, Block3<A>>,
    resolution: f32,
}

impl<A> OccupancyBlockGridMap3<A>
where
    A: na::Scalar,
    Cell<na::U3>: CellToNdIndex<na::U3, nd::Ix3>,
    na::DefaultAllocator: na::allocator::Allocator<A, na::U3>
        + na::allocator::Allocator<isize, na::U3>,
{
    pub fn new(resolution: f32) -> Self {
        OccupancyBlockGridMap3::<A> {
            blocks: std::collections::HashMap::new(),
            resolution: resolution,
        }
    }

    ///// Sets the value at the given cell.
    //#[inline]
    //pub fn set(&mut self, cell: &Cell3, value: A) {
    //    let block = 
    //}
}
