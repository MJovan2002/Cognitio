use num_traits::Float;

use crate::activations::Activation;

/// Exponential Linear Unit [`Activation<T>`] function
/// ```text
///           alpha*e^x^ ; x < 0
/// elu(x) =  0          ; x = 0
///           x          ; x > 0
/// ```
pub struct ELU<T: Float> {
    alpha: T,
}

impl<T: Float> ELU<T> {
    /// [`ELU<T>`] constructor
    pub const fn new(alpha: T) -> Self {
        Self { alpha }
    }
}

impl<T: Float + From<i32>> Activation<T> for ELU<T> {
    fn activate(&self, x: T) -> T {
        if x > T::zero() {
            x
        } else {
            self.alpha * (x.exp() - T::from(1))
        }
    }

    fn derive(&self, x: T) -> T {
        if x > T::zero() {
            T::from(1)
        } else {
            self.alpha * x.exp()
        }
    }
}
