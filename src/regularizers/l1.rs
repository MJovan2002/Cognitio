use num_traits::Number;
use tensor::{BackendProvider, Tensor};

use crate::regularizers::Regularizer;

pub struct L1<T: Number> {
    alpha: T,
}

impl<T: Number + From<i32>> L1<T> {
    pub fn new(alpha: T) -> Self {
        Self { alpha }
    }

    pub fn one() -> Self {
        Self {
            alpha: T::from(1),
        }
    }
}

impl<T: Number, B: BackendProvider, const N: usize> Regularizer<T, B, N> for L1<T> {
    fn regularization(&self, tensor: &Tensor<T, B, N>) -> T {
        tensor.iter().map(|t| t.abs()).sum::<T>() * self.alpha
    }

    fn derive(&self, tensor: &Tensor<T, B, N>) -> Tensor<T, B, N> {
        tensor.shape().clone().into_tensor(|i| tensor[i].signum() * self.alpha)
        // let mut t = Tensor::zero(tensor.get_shape().clone());
        // t.iter_mut()
        //     .zip(tensor.iter())
        //     .for_each(|(a, b)| *a = b.signum() * self.alpha);
        // t
    }
}
