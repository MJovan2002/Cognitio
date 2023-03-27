use num_traits::Float;
use tensor::{BackendProvider, Tensor};
use void::Void;

use crate::optimizers::Optimizer;

const EPSILON: f64 = 0.00000001;

pub struct AdaMax<G> {
    moment_velocity: Option<(G, G)>,
    beta1: f64,
    beta2: f64,
    learning_rate: f64,
    time: u64,
}

impl<G> AdaMax<G> {
    pub fn new(beta1: f64, beta2: f64, learning_rate: f64) -> Self {
        Self { moment_velocity: None, beta1, beta2, learning_rate, time: 0 }
    }
}

impl<G: Iterable + Zero> Optimizer<G> for AdaMax<G> {
    fn gradients_to_deltas(&mut self, mut gradients: G) -> Option<G> {
        self.time += 1;
        let (mut m, mut v) = self.moment_velocity.take().unwrap_or((gradients.zero(), gradients.zero()));
        gradients.iterate(&mut m, &mut v, self.learning_rate, self.beta1, self.beta2, self.time);
        self.moment_velocity = Some((m, v));
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
    fn iterate(&mut self, m: &mut Self, v: &mut Self, learning_rate: f64, beta1: f64, beta2: f64, time: u64);
}

impl Iterable for [Void; 0] {
    fn iterate(&mut self, _: &mut Self, _: &mut Self, _: f64, _: f64, _: f64, _: u64) {}
}

impl<T: Float + From<f64> + Ord, B: BackendProvider, const N: usize> Iterable for Tensor<T, B, N> {
    fn iterate(&mut self, m: &mut Self, v: &mut Self, learning_rate: f64, beta1: f64, beta2: f64, time: u64) {
        self.iter_mut().zip(m.iter_mut().zip(v.iter_mut()))
            .for_each(|(t, (m, v))| {
                *m = T::from(beta1) * *m + T::from(1. - beta1) * *t;
                *v = (T::from(beta2) * *v).max(t.abs());
                let m_ = *m / T::from(1. - beta1.powi(time as i32));
                let v_ = *v / T::from(1. - beta2.powi(time as i32));
                *t = m_ * T::from(learning_rate) / (T::from(EPSILON) + v_.powf(T::from(0.5)))
            })
    }
}

impl<T: Float + From<f64> + Ord, B: BackendProvider, const N: usize, const M: usize> Iterable for [Tensor<T, B, N>; M] {
    fn iterate(&mut self, m: &mut Self, v: &mut Self, learning_rate: f64, beta1: f64, beta2: f64, time: u64) {
        self.into_iter().zip(m.into_iter().zip(v.iter_mut()))
            .for_each(|(t, (m, v))| t.iterate(m, v, learning_rate, beta1, beta2, time))
    }
}

impl<A: Iterable, B: Iterable> Iterable for (A, B) {
    fn iterate(&mut self, m: &mut Self, v: &mut Self, learning_rate: f64, beta1: f64, beta2: f64, time: u64) {
        self.0.iterate(&mut m.0, &mut v.0, learning_rate, beta1, beta2, time);
        self.1.iterate(&mut m.1, &mut v.1, learning_rate, beta1, beta2, time);
    }
}