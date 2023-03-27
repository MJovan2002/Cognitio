use num_traits::Float;

use crate::activations::Activation;

pub struct ReLU<T: Float> {
    alpha: T,
}

impl<T: Float> ReLU<T> {
    pub fn new(alpha: T) -> Self {
        Self { alpha }
    }

    pub fn zero() -> Self {
        Self::new(T::zero())
    }
}

impl<T: Float + From<i32>> Activation<T> for ReLU<T> {
    fn activate(&self, x: T) -> T {
        if x > T::zero() {
            x
        } else {
            self.alpha * x
        }
    }

    fn derive(&self, x: T) -> T {
        if x > T::zero() {
            T::from(1)
        } else {
            self.alpha
        }
    }
}
