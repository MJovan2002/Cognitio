use std::ops::AddAssign;
use tensor::{BackendProvider, Tensor};
use void::Void;

use crate::optimizers::Optimizer;

pub struct MiniBatch<O, G> {
    size: usize,
    inner: O,
    position: usize,
    state: Option<G>,
}

impl<O, G> MiniBatch<O, G> {
    pub fn new(size: usize, inner: O) -> Self {
        Self {
            size,
            inner,
            position: 0,
            state: None,
        }
    }
}

impl<O: Optimizer<G>, G: Combinable> Optimizer<G> for MiniBatch<O, G> {
    fn gradients_to_deltas(&mut self, gradients: G) -> Option<G> {
        let deltas = self.inner.gradients_to_deltas(gradients)?;

        self.state = Some(match self.state.take() {
            None => deltas,
            Some(state) => Combinable::combine(state, deltas),
        });

        self.position = (self.position + 1) % self.size;

        if self.position == 0 {
            self.state.take()
        } else {
            None
        }
    }
}

pub trait IntoMiniBatch<G>: Sized {
    fn batch(self, size: usize) -> MiniBatch<Self, G> {
        MiniBatch::new(size, self)
    }
}

impl<O: Optimizer<G>, G> IntoMiniBatch<G> for O {}

trait Combinable {
    fn combine(a: Self, b: Self) -> Self;
}

impl<T: AddAssign<T>, B: BackendProvider, const N: usize> Combinable for Tensor<T, B, N> {
    fn combine(mut a: Self, b: Self) -> Self {
        a += b;
        a
    }
}

impl<T: AddAssign<T>, B: BackendProvider, const N: usize> Combinable for [Tensor<T, B, N>; N] {
    fn combine(a: Self, b: Self) -> Self {
        a.zip(b).map(|(mut a, b)| {
            a += b;
            a
        })
    }
}

impl Combinable for [Void; 0] {
    fn combine(_: Self, _: Self) -> Self {
        []
    }
}

impl<A: Combinable, B: Combinable> Combinable for (A, B) {
    fn combine(a: Self, b: Self) -> Self {
        (Combinable::combine(a.0, b.0), Combinable::combine(a.1, b.1))
    }
}
