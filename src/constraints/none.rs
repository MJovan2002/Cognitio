use std::marker::PhantomData;

use crate::constraints::Constraint;

pub struct None<T> {
    _marker: PhantomData<T>,
}

impl<T> None<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Constraint<T> for None<T> {
    fn constrain(&self, t: T) -> T {
        t
    }
}
