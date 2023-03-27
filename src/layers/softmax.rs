use std::{
    cmp::Ordering,
    marker::PhantomData,
};

use num_traits::Float;
use tensor::{BackendProvider, Shape, Tensor};
use void::Void;

use crate::{
    layers::{
        Layer,
        LayerBuilder,
    },
};
use crate::data::{Package, Uninitialized};
use crate::data::FromRef;

pub struct SoftMax<T, B: BackendProvider, const N: usize> {
    shape: Shape<N>,
    _maker: PhantomData<(T, B)>,
}

impl<T, B: BackendProvider, const N: usize> SoftMax<T, B, N> {
    fn new(shape: Shape<N>) -> Self {
        Self {
            shape,
            _maker: PhantomData,
        }
    }
}

impl<T: Float, B: BackendProvider<Backend<T>: Clone>, const N: usize> Layer for SoftMax<T, B, N> {
    type Input = Tensor<T, B, N>;
    type ReverseInput = Tensor<T, B, N>;
    type Internal = [Void; 0];
    type Output = Tensor<T, B, N>;
    type ReverseOutput = Tensor<T, B, N>;

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
        &self.shape
    }

    fn feed_forward(
        &self,
        input: Self::Input,
    ) -> Self::Output {
        assert_eq!(input.shape(), &self.shape);

        let mut output = input;
        let max = output
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(T::zero());
        output.iter_mut().for_each(|t| *t = (*t - max).exp());
        let s = output.iter().copied().sum::<T>();
        output.iter_mut().for_each(|t| *t /= s);
        output
    }

    fn back_propagate(
        &self,
        input: Self::Input,
    ) -> (
        Self::Output,
        Self::Computation<'_>,
    ) {
        let output = self.feed_forward(input);
        (
            output.clone(),
            move |mut output_d| {
                assert_eq!(output_d.shape(), &self.shape);
                output_d
                    .iter_mut()
                    .zip(output.iter().copied())
                    .for_each(|(a, b)| *a *= b);
                let sum = output_d.iter().copied().sum::<T>();
                let input_d = output.shape().clone().into_tensor(|i| output_d[i] - sum * output[i]);
                (input_d, [])
            }
        )
    }

    fn update(&mut self, []: &Self::Internal) {}
}

builder::builder! {
    pub struct Builder<(T), (B), const N: usize> {}
}

impl<T: Float, B: BackendProvider<Backend<T>: Clone>, const N: usize> LayerBuilder for Builder<T, B, usizeContainer<N>> {
    type Layer = SoftMax<T, B, N>;

    fn build(self, input_shape: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Self::Layer::new(input_shape)
    }
}
