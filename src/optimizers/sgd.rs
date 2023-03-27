use std::ops::AddAssign;
use num_traits::Number;
use tensor::{BackendProvider, SliceIndex, Tensor};
use void::Void;

use crate::{
    optimizers::Optimizer,
    schedules::LearningRateSchedule,
};

pub struct SGD<S, G> {
    learning_rate: S,
    momentum: G,
}

impl<S> SGD<S, ()> {
    pub fn no_momentum(learning_rate: S) -> Self {
        Self {
            learning_rate,
            momentum: (),
        }
    }
}

impl<S0, S1, G> SGD<S0, (S1, Option<G>)> {
    pub fn momentum(learning_rate: S0, momentum: S1) -> Self {
        Self {
            learning_rate,
            momentum: (momentum, None),
        }
    }
}

impl<S: LearningRateSchedule<f64>, G: Iterable> Optimizer<G> for SGD<S, ()> {
    fn gradients_to_deltas(&mut self, gradients: G) -> Option<G> {
        Some(gradients.modify(&mut Comp {
            alpha: &mut self.learning_rate,
        }))
    }
}

impl<S0: LearningRateSchedule<f64>, S1: LearningRateSchedule<f64>, G: Iterable + Mergeable + Clone> Optimizer<G> for SGD<S0, (S1, Option<G>)> {
    fn gradients_to_deltas(&mut self, gradients: G) -> Option<G> {
        let mut gradients = gradients.modify(&mut Comp {
            alpha: &mut self.learning_rate,
        });

        self.momentum.1 = Some(match self.momentum.1.take() {
            None => gradients.clone(),
            Some(deltas) => {
                let mut deltas = deltas.modify(&mut Comp {
                    alpha: &mut self.momentum.0,
                });
                Mergeable::merge(&mut deltas, &mut gradients);
                deltas
            }
        });
        Some(gradients)
    }
}

struct Comp<'s, S> {
    alpha: &'s mut S,
}

impl<'s, S: LearningRateSchedule<f64>> Modifier for Comp<'s, S> {
    fn modify<T: Number + From<f64>, B: BackendProvider, const N: usize>(
        &mut self,
        mut input: Tensor<T, B, N>,
    ) -> Tensor<T, B, N> {
        let alpha = T::from(self.alpha.next());
        input *= alpha;
        input
    }
}

trait Iterable {
    fn modify<C: Modifier>(self, c: &mut C) -> Self;
}

impl<A: Number + From<f64>, B: BackendProvider, const N: usize> Iterable for Tensor<A, B, N> {
    fn modify<C: Modifier>(self, c: &mut C) -> Self {
        c.modify(self)
    }
}

impl<A: Number + From<f64>, B: BackendProvider, const N: usize, const M: usize> Iterable for [Tensor<A, B, N>; M] {
    fn modify<C: Modifier>(self, c: &mut C) -> Self {
        self.map(|t| c.modify(t))
    }
}

impl Iterable for [Void; 0] {
    fn modify<C: Modifier>(self, _: &mut C) -> Self {
        []
    }
}

impl<A: Iterable, B: Iterable> Iterable for (A, B) {
    fn modify<C: Modifier>(self, c: &mut C) -> Self {
        (self.0.modify(c), self.1.modify(c))
    }
}

trait Modifier {
    fn modify<T: Number + From<f64>, B: BackendProvider, const N: usize>(&mut self, input: Tensor<T, B, N>) -> Tensor<T, B, N>;
}

trait Mergeable {
    fn merge(a: &mut Self, b: &mut Self);
}

impl Mergeable for [Void; 0]{
    fn merge([]: &mut Self, []: &mut Self) {}
}

impl<T: for<'s> AddAssign<&'s mut T>, B: BackendProvider, const N: usize> Mergeable for Tensor<T, B, N> {
    fn merge(a: &mut Self, b: &mut Self) {
        *a += b.slice_mut(SliceIndex::full()).unwrap();
    }
}

impl<T: for<'s> AddAssign<&'s mut T>, B: BackendProvider, const N: usize, const M: usize> Mergeable for [Tensor<T, B, N>; M] {
    fn merge(a: &mut Self, b: &mut Self) {
        a.into_iter().zip(b.into_iter()).for_each(|(a, b)| Mergeable::merge(a, b))
    }
}

impl<A: Mergeable, B: Mergeable> Mergeable for (A, B) {
    fn merge(a: &mut Self, b: &mut Self) {
        Mergeable::merge(&mut a.0, &mut b.0);
        Mergeable::merge(&mut a.1, &mut b.1);
    }
}
