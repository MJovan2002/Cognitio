//! Trait that defines [`Activation<T>`] functions

pub mod elu;
pub mod exp;
pub mod identity;
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod softplus;
pub mod softsign;
pub mod swish;
pub mod tanh;

/// Type that activates it's input
pub trait Activation<T> {
    fn activate(&self, x: T) -> T;

    fn derive(&self, x: T) -> T;
}
