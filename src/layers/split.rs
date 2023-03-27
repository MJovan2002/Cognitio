use std::{
    array,
    marker::PhantomData,
};

use num_traits::Number;
use tensor::{BackendProvider, Shape, Tensor};
use void::Void;
use builder::builder;

use crate::data::{Package, FromRef, Uninitialized};
use crate::layers::{Layer, LayerBuilder};

pub struct Split<T, B: BackendProvider, const N: usize, const M: usize> {
    shape: Shape<N>,
    _maker: PhantomData<(T, B)>,
}

impl<T, B: BackendProvider, const N: usize, const M: usize> Split<T, B, N, M> {
    fn new(shape: Shape<N>) -> Self {
        Self {
            shape,
            _maker: PhantomData,
        }
    }
}

impl<T: Number, B: BackendProvider<Backend<T>: Clone>, const N: usize, const M: usize> Layer for Split<T, B, N, M> {
    type Input = Tensor<T, B, N>;
    type ReverseInput = Tensor<T, B, N>;
    type Internal = [Void; 0];
    type Output = [Tensor<T, B, N>; M];
    type ReverseOutput = [Tensor<T, B, N>; M];

    type Computation<'s> = impl FnOnce(
        Self::ReverseOutput
    ) -> (
        Self::ReverseInput,
        Self::Internal,
    ) + 's where Self: 's;

    fn input_shapes(&self) -> <<Self::Input as Package>::Shapes as FromRef>::Ref<'_> {
        &self.shape
    }

    fn output_shapes(&self) -> <<Self::Output as Package>::Shapes as FromRef>::Ref<'_> {
        [&self.shape; M]
    }

    fn feed_forward(
        &self,
        input: Self::Input,
    ) -> Self::Output {
        assert_eq!(input.shape(), &self.shape);

        array::from_fn(|_| input.clone())
    }

    fn back_propagate(
        &self,
        input: Self::Input,
    ) -> (
        Self::Output,
        Self::Computation<'_>,
    ) {
        (
            self.feed_forward(input),
            |output_d| {
                assert!(output_d.iter().all(|o| o.shape() == &self.shape));
                (output_d.into_iter().fold(
                    Tensor::new(T::zero(), self.shape.clone()),
                    |acc, t| acc + t
                ), [])
            }
        )
    }

    fn update(&mut self, []: &Self::Internal) {}
}

builder! {
    pub struct Builder<(T), (B), const N: usize, const M: usize> {}
}

impl<T: Number, B: BackendProvider<Backend<T>: Clone>, const N: usize, const M: usize> LayerBuilder for Builder<T, B, usizeContainer<N>, usizeContainer<M>> {
    type Layer = Split<T, B, N, M>;

    fn build(self, input_shapes: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Self::Layer::new(input_shapes.clone())
    }
}
