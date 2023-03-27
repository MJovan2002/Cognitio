use std::{
    array,
    marker::PhantomData,
    ops::AddAssign,
};

use num_traits::Number;
use tensor::{BackendProvider, Shape, SliceIndex, Tensor};
use void::Void;

use crate::{
    data::{Package, FromRef, Uninitialized},
    layers::{Layer, LayerBuilder},
};

pub struct Add<T, B, const N: usize, const M: usize> {
    shape: Shape<N>,
    _marker: PhantomData<(T, B)>,
}

impl<T, B, const N: usize, const M: usize> Add<T, B, N, M> {
    pub fn new(shape: Shape<N>) -> Self {
        Self { shape, _marker: PhantomData }
    }
}

impl<T: Number + for<'s> AddAssign<&'s T>, B: BackendProvider<Backend<T>: Clone>, const N: usize, const M: usize> Layer for Add<T, B, N, M> {
    type Input = [Tensor<T, B, N>; M];
    type ReverseInput = [Tensor<T, B, N>; M];
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
        [&self.shape; M]
    }

    fn output_shapes(&self) -> <<Self::Output as Package>::Shapes as FromRef>::Ref<'_> {
        &self.shape
    }

    fn feed_forward(
        &self,
        input: Self::Input,
    ) -> Self::Output {
        input.into_iter().map(|t| {
            assert_eq!(t.shape_refs(), &self.shape);
            t
        }).fold(Tensor::new(T::zero(), self.shape.clone()), |mut a, t| {
            a += t.slice(SliceIndex::full()).unwrap();
            a
        })
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
                assert_eq!(output_d.shape(), &self.shape);
                (array::from_fn(|_| output_d.clone()), [])
            }
        )
    }

    fn update(&mut self, []: &Self::Internal) {}
}

builder::builder! {
    pub struct Builder<(T), (B), const N: usize, const M: usize> {}
}

impl<T: Number + for<'s> AddAssign<&'s T>, B: BackendProvider<Backend<T>: Clone>, const N: usize, const M: usize> LayerBuilder for Builder<T, B, usizeContainer<N>, usizeContainer<M>> {
    type Layer = Add<T, B, N, M>;

    fn build(self, input_shapes: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Self::Layer::new(input_shapes[0].clone())
    }
}
