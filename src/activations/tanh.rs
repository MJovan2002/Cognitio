use num_traits::Float;
use std::marker::PhantomData;

use crate::activations::Activation;

pub struct Tanh<T> {
    _marker: PhantomData<T>,
}

impl<T> Tanh<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for Tanh<T>{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + From<i32>> Activation<T> for Tanh<T> {
    fn activate(&self, x: T) -> T {
        x.tanh()
    }

    fn derive(&self, x: T) -> T {
        T::from(1) / x.cosh().powi(2)
    }
}
