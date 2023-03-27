use num_traits::Float;
use std::marker::PhantomData;

use crate::activations::Activation;

/// Exponential [`Activation<T>`] function
/// ```text
/// exp(x) =  e^x
/// ```
pub struct EXP<T: Float> {
    _marker: PhantomData<T>,
}

impl<T: Float> EXP<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Float> Activation<T> for EXP<T> {
    fn activate(&self, x: T) -> T {
        x.exp()
    }

    fn derive(&self, x: T) -> T {
        x.exp()
    }
}
