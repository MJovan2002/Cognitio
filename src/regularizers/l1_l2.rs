use num_traits::Number;
use tensor::{BackendProvider, Tensor};
use crate::regularizers::Regularizer;

pub struct L1L2<T> {
    l1: T,
    l2: T,
}

impl<T> L1L2<T> {
    pub fn new(l1: T, l2: T) -> Self {
        Self {
            l1,
            l2,
        }
    }
}

impl<T: Number + From<i32>, B: BackendProvider, const N: usize> Regularizer<T, B, N> for L1L2<T> {
    fn regularization(&self, tensor: &Tensor<T, B, N>) -> T {
        tensor.iter().map(|t| t.abs() * self.l1 + *t * *t * self.l2).sum::<T>()
    }

    fn derive(&self, tensor: &Tensor<T, B, N>) -> Tensor<T, B, N> {
        tensor.shape().clone().into_tensor(|i| tensor[i].signum() * self.l1 + T::from(2) * tensor[i] * self.l2)
    }
}