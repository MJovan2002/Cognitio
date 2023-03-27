use std::{
    marker::PhantomData,
    ops::Mul,
};

use num_traits::Number;
use tensor::{BackendProvider, Shape, Tensor};
use void::Void;

use crate::{
    data::{
        Package,
        FromRef,
        Uninitialized,
    },
    layers::{Layer, LayerBuilder},
};

pub struct Dot<T, B> {
    shape: Shape<1>,
    // axes: (usize, usize),
    _marker: PhantomData<(T, B)>,
}

impl<T: Number, B: BackendProvider> Dot<T, B> where for<'s> &'s T: Mul<&'s T, Output=T> {
    pub fn new(shape: Shape<1>/*, axes: (usize, usize)*/) -> Self {
        Self {
            shape,
            // axes,
            _marker: PhantomData,
        }
    }

    fn feed_forward(&self, a: &Tensor<T, B, 1>, b: &Tensor<T, B, 1>) -> Tensor<T, B, 0> {
        let t = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<T>();
        todo!()
    }
}

impl<T: Number, B: BackendProvider> Layer for Dot<T, B> where for<'s> &'s T: Mul<&'s T, Output=T> {
    type Input = [Tensor<T, B, 1>; 2];
    type ReverseInput = [Tensor<T, B, 1>; 2];
    type Internal = [Void; 0];
    type Output = Tensor<T, B, 0>;
    type ReverseOutput = Tensor<T, B, 0>;
    type Computation<'s> = impl FnOnce(
        Self::ReverseOutput
    ) -> (
        Self::ReverseInput,
        Self::Internal
    ) + 's where Self: 's;

    fn input_shapes(&self) -> <<Self::Input as Package>::Shapes as FromRef>::Ref<'_> {
        [&self.shape; 2]
    }

    fn output_shapes(&self) -> <<Self::Output as Package>::Shapes as FromRef>::Ref<'_> {
        const ZERO_SHAPE: Shape<0> = Shape::zero();
        &ZERO_SHAPE
    }

    fn feed_forward(&self, [a, b]: Self::Input) -> Self::Output {
        self.feed_forward(&a, &b)
    }

    fn back_propagate(&self, [mut a, mut b]: Self::Input) -> (Self::Output, Self::Computation<'_>) {
        (self.feed_forward(&a, &b), |output_d| {
            let output_d = *output_d;
            a *= output_d;
            b *= output_d;
            ([b, a], [])
        })
    }

    fn update(&mut self, []: &Self::Internal) {}
}

builder::builder! {
    pub struct Builder<(T), (B)> {
        // axis0: A0,
        // axis1: A1,
    }
}

impl<T: Number, B: BackendProvider> LayerBuilder for Builder<T, B> where for<'s> &'s T: Mul<&'s T, Output=T> {
    type Layer = Dot<T, B>;

    fn build(self, [shape0, shape1]: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        assert_eq!(shape0, shape1);
        Self::Layer::new(
            shape0,
            // (self.axis0, self.axis1)
        )
    }
}
