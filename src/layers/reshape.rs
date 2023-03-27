use std::marker::PhantomData;
use tensor::{BackendProvider, Shape, Tensor};
use void::Void;

use crate::{
    data::{Package, FromRef, Uninitialized},
    layers::{Layer, LayerBuilder},
};

pub struct Reshape<T, B, const N: usize, const M: usize> {
    input_shape: Shape<N>,
    output_shape: Shape<M>,
    _marker: PhantomData<(T, B)>,
}

impl<T, B, const N: usize, const M: usize> Reshape<T, B, N, M> {
    fn new(input_shape: Shape<N>, output_shape: Shape<M>) -> Self {
        Self {
            input_shape,
            output_shape,
            _marker: PhantomData,
        }
    }
}

impl<T, B: BackendProvider, const N: usize, const M: usize> Layer for Reshape<T, B, N, M> {
    type Input = Tensor<T, B, N>;
    type ReverseInput = Tensor<T, B, N>;
    type Internal = [Void; 0];
    type Output = Tensor<T, B, M>;
    type ReverseOutput = Tensor<T, B, M>;
    type Computation<'s> = impl FnOnce(
        Self::ReverseOutput
    ) -> (
        Self::ReverseInput,
        Self::Internal
    ) + 's where Self: 's;

    fn input_shapes(&self) -> <<Self::Input as Package>::Shapes as FromRef>::Ref<'_> {
        &self.input_shape
    }

    fn output_shapes(&self) -> <<Self::Output as Package>::Shapes as FromRef>::Ref<'_> {
        &self.output_shape
    }

    fn feed_forward(&self, input: Self::Input) -> Self::Output {
        todo!()
    }

    fn back_propagate(&self, input: Self::Input) -> (Self::Output, Self::Computation<'_>) {
        (
            self.feed_forward(input),
            |output_d| {
                assert_eq!(output_d.shape(), &self.output_shape);
                todo!()
            }
        )
    }

    fn update(&mut self, []: &Self::Internal) {}
}

builder::builder! {
    pub struct Builder<(T), (B), const N: usize, const M: usize> {
        output_shape: S,
    }
}

impl<T, B: BackendProvider, const N: usize, const M: usize> LayerBuilder for Builder<Shape<M>, T, B, usizeContainer<N>, usizeContainer<M>> {
    type Layer = Reshape<T, B, N, M>;

    fn build(self, input_shape: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Self::Layer::new(input_shape, self.output_shape)
    }
}
