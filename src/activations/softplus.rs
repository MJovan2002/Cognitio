use num_traits::Float;
use std::marker::PhantomData;

use crate::activations::Activation;

pub struct SoftPlus<T> {
    _marker: PhantomData<T>,
}

impl<T> SoftPlus<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for SoftPlus<T>{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + From<i32>> Activation<T> for SoftPlus<T> {
    fn activate(&self, x: T) -> T {
        (x.exp() + T::from(1)).ln()
    }

    fn derive(&self, x: T) -> T {
        T::from(1) / (T::from(1) + (-x).exp())
    }
}
