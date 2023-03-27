use std::marker::PhantomData;

use num_traits::Float;

use crate::activations::Activation;

pub struct Sigmoid<T> {
    _marker: PhantomData<T>,
}

impl<T> Sigmoid<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for Sigmoid<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + From<i32>> Activation<T> for Sigmoid<T> {
    fn activate(&self, x: T) -> T {
        T::from(1) / (T::from(1) + (-x).exp())
    }

    fn derive(&self, x: T) -> T {
        T::from(1) / (x.exp() + T::from(2) + (-x).exp())
    }
}
