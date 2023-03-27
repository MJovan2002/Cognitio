use std::marker::PhantomData;

use num_traits::Float;

use crate::activations::Activation;

/// Identity [`Activation<T>`] function
/// ```text
/// identity(x) =  x
/// ```
pub struct Identity<T> {
    _marker: PhantomData<T>,
}

impl<T> Identity<T> {
    /// [`Identity<T>`] constructor
    pub const fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<T> const Default for Identity<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + From<i32>> Activation<T> for Identity<T> {
    fn activate(&self, x: T) -> T {
        x
    }

    fn derive(&self, _: T) -> T {
        T::from(1)
    }
}
