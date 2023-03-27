use tensor::{BackendProvider, Tensor};
use num_traits::Number;
use crate::data::{Initialized, Uninitialized};
use crate::regularizers::none::None as NoneReg;

pub mod l1;
pub mod l2;
pub mod l1_l2;
pub mod none;

pub trait Regularizer<T, B: BackendProvider, const N: usize> {
    fn regularization(&self, tensor: &Tensor<T, B, N>) -> T;

    fn derive(&self, tensor: &Tensor<T, B, N>) -> Tensor<T, B, N>;
}

pub trait IntoRegularizer<T, B: BackendProvider, const N: usize> {
    type Regularizer: Regularizer<T, B, N>;

    fn into_regularizer(self) -> Self::Regularizer;
}

impl<T: Number, B: BackendProvider, const N: usize> IntoRegularizer<T, B, N> for Uninitialized {
    type Regularizer = NoneReg<T>;

    fn into_regularizer(self) -> Self::Regularizer {
        NoneReg::new()
    }
}

impl<T, B: BackendProvider, const N: usize, R: Regularizer<T, B, N> + Initialized> IntoRegularizer<T, B, N> for R {
    type Regularizer = R;

    fn into_regularizer(self) -> Self::Regularizer {
        self
    }
}
