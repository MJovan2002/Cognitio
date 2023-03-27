use rand::distributions::Distribution;
use rand_distr::{
    Normal as NormalDist,
    StandardNormal,
    Uniform as UniformDist,
    uniform::SampleUniform,
    num_traits::Float,
};
use tensor::{BackendProvider, Shape, Tensor};

use crate::initializers::Initializer;

pub trait IntoDist {
    type T;
    type Distribution: Distribution<Self::T>;

    fn resolve(&mut self, shape: &Shape<2>) -> Self::Distribution;
}

pub struct Normal<T>(T, Mode);

impl<T: Float> IntoDist for Normal<T> where StandardNormal: Distribution<T> {
    type T = T;
    type Distribution = NormalDist<Self::T>;

    fn resolve(&mut self, shape: &Shape<2>) -> Self::Distribution {
        let n = self.1.resolve(shape);
        NormalDist::new(T::zero(), (self.0 / T::from(n).unwrap()).sqrt()).unwrap()
    }
}

pub struct Uniform<T>(T, Mode);

impl<T: SampleUniform + Clone + Float> IntoDist for Uniform<T> {
    type T = T;
    type Distribution = UniformDist<T>;

    fn resolve(&mut self, shape: &Shape<2>) -> Self::Distribution {
        let limit = T::from(self.1.resolve(shape)).unwrap();
        UniformDist::new(-limit, limit)
    }
}

pub struct VarianceScaling<D: IntoDist> {
    distribution: D,
}

impl<T: Float> VarianceScaling<Normal<T>> where StandardNormal: Distribution<T> {
    pub const fn normal(scale: T, mode: Mode) -> Self {
        Self { distribution: Normal(scale, mode) }
    }

    pub fn glorot() -> Self {
        Self::normal(T::one(), Mode::FanAvg)
    }

    pub fn he() -> Self {
        Self::normal(T::from(2).unwrap(), Mode::FanIn)
    }

    pub fn lecun() -> Self {
        Self::normal(T::one(), Mode::FanIn)
    }
}

impl<T: SampleUniform + Float> VarianceScaling<Uniform<T>> {
    pub const fn uniform(scale: T, mode: Mode) -> Self {
        Self { distribution: Uniform(scale, mode) }
    }

    pub fn glorot() -> Self {
        Self::uniform(T::one(), Mode::FanAvg)
    }

    pub fn he() -> Self {
        Self::uniform(T::from(2).unwrap(), Mode::FanIn)
    }

    pub fn lecun() -> Self {
        Self::uniform(T::one(), Mode::FanIn)
    }
}

pub enum Mode {
    FanIn,
    FanOut,
    FanAvg,
}

impl Mode {
    fn resolve(&self, shape: &Shape<2>) -> f64 {
        match self {
            Mode::FanIn => shape[0] as f64,
            Mode::FanOut => shape[1] as f64,
            Mode::FanAvg => (shape[0] + shape[1]) as f64,
        }
    }
}

impl<B: BackendProvider, D: IntoDist> Initializer<D::T, B, 2> for VarianceScaling<D> {
    fn initialize(&mut self, shape: Shape<2>) -> Tensor<D::T, B, 2> {
        let d = self.distribution.resolve(&shape);
        shape.into_tensor(|_| d.sample(&mut rand::thread_rng()))
    }
}
