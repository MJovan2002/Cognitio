use std::marker::PhantomData;

use num_traits::Number;
use tensor::{BackendProvider, Tensor};

use crate::{regularizers::Regularizer};

pub struct None<T> {
    _marker: PhantomData<T>,
}

impl<T> None<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Number, B: BackendProvider, const N: usize> Regularizer<T, B, N> for None<T> {
    fn regularization(&self, _: &Tensor<T, B, N>) -> T {
        T::zero()
    }

    fn derive(&self, tensor: &Tensor<T, B, N>) -> Tensor<T, B, N> {
        todo!()
    }
}
