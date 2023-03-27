use std::ops::{Mul, Sub};
use tensor::{BackendProvider, Tensor};
use void::Void;

use crate::losses::Loss;

pub struct Square {
    _private: (),
}

impl Square {
    pub const fn new() -> Self {
        Self { _private: () }
    }
}

impl<I: Zip> Loss<I, I> for Square {
    // fn loss(&self, predicted: &[Tensor<T>; N], expected: &[Tensor<T>; N]) -> T {
    //     predicted
    //         .iter()
    //         .zip(expected.iter())
    //         .map(|(a, b)| a.iter().copied().zip(b.iter().copied()).map(|(a, b)| (a - b) * (a - b)).sum())
    //         .sum()
    // }

    fn derive(&self, predicted: &I, expected: &I) -> I {
        I::zip(predicted, expected)
    }
}

trait Zip {
    fn zip(a: &Self, b: &Self) -> Self;
}

impl Zip for [Void; 0] {
    fn zip([]: &Self, []: &Self) -> Self {
        []
    }
}

impl<T: From<i32> + Clone + Mul<Output=T>, B: BackendProvider, const N: usize> Zip for Tensor<T, B, N> where for<'s> &'s T: Sub<&'s T, Output=T> {
    fn zip(a: &Self, b: &Self) -> Self {
        (a - b) * T::from(2)
    }
}

impl<T: From<i32> + Clone + Mul<Output=T>, B: BackendProvider, const N: usize, const L: usize> Zip for [Tensor<T, B, N>; L] where for<'s> &'s T: Sub<&'s T, Output=T> {
    fn zip(a: &Self, b: &Self) -> Self {
        a.each_ref().zip(b.each_ref()).map(|(a, b)| Zip::zip(a, b))
    }
}

impl<T: Zip, U: Zip> Zip for (T, U) {
    fn zip(a: &Self, b: &Self) -> Self {
        (Zip::zip(&a.0, &b.0), Zip::zip(&a.1, &b.1))
    }
}
