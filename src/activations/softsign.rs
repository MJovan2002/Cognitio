use num_traits::Float;
use std::marker::PhantomData;

use crate::activations::Activation;

pub struct SoftSign<T> {
    _marker: PhantomData<T>,
}

impl<T> SoftSign<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for SoftSign<T>{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + From<i32>> Activation<T> for SoftSign<T> {
    fn activate(&self, x: T) -> T {
        x / (T::from(1) + x.abs())
    }

    fn derive(&self, x: T) -> T {
        T::from(1) / (T::from(1) + x.abs()).powi(2)
    }
}
