use std::marker::PhantomData;

use num_traits::Number;

use crate::constraints::Constraint;

pub struct Positive<T: Number> {
    _marker: PhantomData<T>,
}

impl<T: Number> Positive<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Number> Constraint<T> for Positive<T> {
    fn constrain(&self, t: T) -> T {
        if t < T::zero() {
            T::zero()
        } else {
            t
        }
    }
}
