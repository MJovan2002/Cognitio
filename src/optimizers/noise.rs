use std::{
    ops::MulAssign,
    iter,
};

use tensor::{BackendProvider, Tensor};
use rand_distr::Normal;
use rand::Rng;
use void::Void;

use crate::optimizers::Optimizer;

pub struct Noisy<O> {
    optimizer: O,
    t: u64,
    alpha: f64,
    gamma: f64,
}

impl<O> Noisy<O> {
    fn get_distr(&mut self) -> Normal<f64> {
        self.t += 1;
        Normal::new(0., (self.alpha / (self.t as f64).powf(self.gamma)).sqrt()).unwrap()
    }
}

impl<G: Iterable, O: Optimizer<G>> Optimizer<G> for Noisy<O> {
    fn gradients_to_deltas(&mut self, mut gradients: G) -> Option<G> {
        let d = self.get_distr();
        gradients.iterate(d);
        self.optimizer.gradients_to_deltas(gradients)
    }
}

pub trait AddNoise<G>: Sized {
    fn noisy(self, alpha: f64, gamma: f64) -> Noisy<Self> {
        Noisy {
            optimizer: self,
            t: 0,
            alpha,
            gamma,
        }
    }
}

impl<G, O: Optimizer<G>> AddNoise<G> for O {}

trait Iterable {
    fn iterate(&mut self, d: Normal<f64>);
}

impl<T: MulAssign<T> + From<f64>, B: BackendProvider, const N: usize> Iterable for Tensor<T, B, N> {
    fn iterate(&mut self, d: Normal<f64>) {
        self.iter_mut().zip(iter::from_fn(|| Some(rand::thread_rng().sample(d)))).for_each(|(a, b)| *a *= T::from(b))
    }
}

impl<T: MulAssign<T> + From<f64>, B: BackendProvider, const N: usize> Iterable for [Tensor<T, B, N>; N] {
    fn iterate(&mut self, d: Normal<f64>) {
        self.into_iter().for_each(|t| t.iterate(d))
    }
}

impl Iterable for [Void; 0] {
    fn iterate(&mut self, _: Normal<f64>) {}
}

impl<A: Iterable, B: Iterable> Iterable for (A, B) {
    fn iterate(&mut self, d: Normal<f64>) {
        self.0.iterate(d.clone());
        self.1.iterate(d);
    }
}
