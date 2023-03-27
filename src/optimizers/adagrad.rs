use num_traits::Float;
use tensor::{BackendProvider, Tensor};
use void::Void;

use crate::optimizers::Optimizer;

const EPSILON: f64 = 0.00000001;

pub struct AdaGrad<G> {
    learning_rate: f64,
    g: Option<G>,
}

impl<G> AdaGrad<G> {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate, g: None }
    }
}

impl<G: Iterable + Zero> Optimizer<G> for AdaGrad<G> {
    fn gradients_to_deltas(&mut self, mut gradients: G) -> Option<G> {
        let mut g = self.g.take().unwrap_or(gradients.zero());
        gradients.iterate(&mut g, self.learning_rate);
        self.g = Some(g);
        Some(gradients)
    }
}

trait Zero {
    fn zero(&self) -> Self;
}

impl Zero for [Void; 0] {
    fn zero(&self) -> Self {
        []
    }
}

impl<T: Float, B: BackendProvider, const N: usize> Zero for Tensor<T, B, N> {
    fn zero(&self) -> Self {
        Self::new(T::zero(), self.shape().clone())
    }
}

impl<T: Float, B: BackendProvider, const N: usize, const M: usize> Zero for [Tensor<T, B, N>; M] {
    fn zero(&self) -> Self {
        self.each_ref().map(|t| t.zero())
    }
}

impl<A: Zero, B: Zero> Zero for (A, B) {
    fn zero(&self) -> Self {
        (self.0.zero(), self.1.zero())
    }
}

trait Iterable: Sized {
    fn iterate(&mut self, g: &mut Self, learning_rate: f64);
}

impl Iterable for [Void; 0] {
    fn iterate(&mut self, _: &mut Self, _: f64) {}
}

impl<T: Float + From<f64>, B: BackendProvider, const N: usize> Iterable for Tensor<T, B, N> {
    fn iterate(&mut self, g: &mut Self, learning_rate: f64) {
        self.iter_mut().zip(g.iter_mut())
            .for_each(|(t, g)| {
                *g += *t * *t;
                *t *= T::from(learning_rate) / (*g + T::from(EPSILON)).powf(T::from(0.5))
            })
    }
}

impl<T: Float + From<f64>, B: BackendProvider, const N: usize, const M: usize> Iterable for [Tensor<T, B, N>; M] {
    fn iterate(&mut self, g: &mut Self, learning_rate: f64) {
        self.into_iter().zip(g.into_iter()).for_each(|(t, g)| t.iterate(g, learning_rate))
    }
}

impl<A: Iterable, B: Iterable> Iterable for (A, B) {
    fn iterate(&mut self, g: &mut Self, learning_rate: f64) {
        self.0.iterate(&mut g.0, learning_rate);
        self.1.iterate(&mut g.1, learning_rate);
    }
}