use tensor::{BackendProvider, Shape, Tensor};
use num_traits::Number;
use crate::data::{Initialized, Uninitialized};

pub mod constant;
pub mod random;
#[cfg(feature = "distributions")]
pub mod variance_scaling;

pub trait Initializer<T, B: BackendProvider, const N: usize> {
    fn initialize(&mut self, shape: Shape<N>) -> Tensor<T, B, N>;
}

pub trait IntoInitializer<T, B: BackendProvider, const N: usize> {
    type Initializer: Initializer<T, B, N>;

    fn into_initializer(self) -> Self::Initializer;
}

impl<T: Number, B: BackendProvider, const N: usize> IntoInitializer<T, B, N> for Uninitialized {
    type Initializer = T;

    fn into_initializer(self) -> Self::Initializer {
        T::zero()
    }
}

impl<T, B: BackendProvider, const N: usize, I: Initializer<T, B, N> + Initialized> IntoInitializer<T, B, N> for I {
    type Initializer = I;

    fn into_initializer(self) -> Self::Initializer {
        self
    }
}
