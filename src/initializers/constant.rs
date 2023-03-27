use num_traits::Number;
use tensor::{BackendProvider, Shape, Tensor};

use crate::initializers::Initializer;

impl<T: Number, B:BackendProvider, const N: usize> Initializer<T, B, N> for T {
    fn initialize(&mut self, shape: Shape<N>) -> Tensor<T, B, N> {
        shape.into_tensor(|_| *self)
    }
}
