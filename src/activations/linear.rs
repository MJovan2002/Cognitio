use num_traits::Number;

use crate::activations::Activation;

pub struct Linear<T: Number> {
    a: T,
    b: T,
}

impl<T: Number + From<i32>> Linear<T> {
    pub fn new(a: T, b: T) -> Self {
        Self { a, b }
    }

    pub fn identity() -> Self {
        Self::new(T::from(1), T::zero())
    }
}

impl<T: Number> Activation<T> for Linear<T> {
    fn activate(&self, x: T) -> T {
        self.a * x + self.b
    }

    fn derive(&self, _: T) -> T {
        self.a
    }
}
