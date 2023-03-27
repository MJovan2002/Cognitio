use num_traits::Float;
use std::marker::PhantomData;

use crate::activations::Activation;

pub struct Swish<T> {
    _marker: PhantomData<T>,
}

impl<T> Swish<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for Swish<T>{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + From<i32>> Activation<T> for Swish<T> {
    fn activate(&self, x: T) -> T {
        x / (T::from(1) + (-x).exp())
    }

    fn derive(&self, x: T) -> T {
        ((T::from(2) * x).exp() + x.exp() + x * x.exp()) / (T::from(1) + x.exp()).powi(2)
    }
}
