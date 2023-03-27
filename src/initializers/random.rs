use tensor::{BackendProvider, Shape, Tensor};
use rand::distributions::Distribution;

#[cfg(feature = "distributions")]
use rand::distributions::uniform::SampleUniform;

#[cfg(feature = "distributions")]
use rand_distr::{
    num_traits::Float,
    Normal,
    StandardNormal,
    NormalError,
    Uniform,
};

use crate::initializers::Initializer;

pub struct Random<D> {
    distribution: D,
}

impl<D> Random<D> {
    pub fn new(distribution: D) -> Self {
        Self { distribution }
    }
}

#[cfg(feature = "distributions")]
impl<T: Float> Random<Normal<T>> where StandardNormal: Distribution<T> {
    pub fn normal(mean: T, std_dev: T) -> Result<Self, NormalError> {
        Ok(Self::new(Normal::new(mean, std_dev)?))
    }
}

#[cfg(feature = "distributions")]
impl<T: SampleUniform> Random<Uniform<T>> {
    pub fn uniform(low: T, high: T) -> Self {
        Self::new(Uniform::new(low, high))
    }
}

impl<T, B: BackendProvider, const N: usize, D: Distribution<T>> Initializer<T, B, N> for Random<D> {
    fn initialize(&mut self, shape: Shape<N>) -> Tensor<T, B, N> {
        shape.into_tensor(|_| self.distribution.sample(&mut rand::thread_rng()))
    }
}