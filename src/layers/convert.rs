use std::marker::PhantomData;

use num_traits::{Float, Number};
use tensor::{BackendProvider, Shape, Tensor};
use void::Void;

use crate::{
    activations::{
        Activation,
        identity::Identity,
    },
    layers::{
        Layer,
        LayerBuilder,
    },
};
use crate::data::{Package, Uninitialized};
use crate::data::FromRef;

pub struct Convert<T, U, B, A, const N: usize> {
    shape: Shape<N>,
    activation: A,
    _marker: PhantomData<(T, U, B)>,
}

impl<T, U, B: BackendProvider, A, const N: usize> Convert<T, U, B, A, N> {
    fn new(shape: Shape<N>, activation: A) -> Self {
        Self {
            shape,
            activation,
            _marker: PhantomData,
        }
    }
}

impl<T, U, B: BackendProvider, A: Activation<U>, const N: usize> Convert<T, U, B, A, N> {
    fn feed_forward<F: FnMut(usize, U)>(&self, input: &Tensor<T, B, N>, mut f: F) -> Tensor<U, B, N> {
        todo!()
        // self.shape.clone().into_tensor(move |i| {
        //     let t = input[i].into();
        //     f(i, self.activation.derive(t));
        //     self.activation.activate(t)
        // })
    }
}

impl<T, U, B: BackendProvider, A: Activation<U>, const N: usize> Layer for Convert<T, U, B, A, N> {
    type Input = Tensor<T, B, N>;
    type ReverseInput = Tensor<T, B, N>;
    type Internal = [Void; 0];
    type Output = Tensor<U, B, N>;
    type ReverseOutput = Tensor<U, B, N>;

    type Computation<'s> = impl FnOnce(
        Self::ReverseOutput
    ) -> (
        Self::ReverseInput,
        Self::Internal
    ) + 's where Self: 's;

    fn input_shapes(&self) -> <<Self::Input as Package>::Shapes as FromRef>::Ref<'_> {
        &self.shape
    }

    fn output_shapes(&self) -> <<Self::Output as Package>::Shapes as FromRef>::Ref<'_> {
        &self.shape
    }

    fn feed_forward(&self, input: Self::Input) -> Self::Output {
        self.feed_forward(&input, |_, _| {})
    }

    fn back_propagate(&self, input: Self::Input) -> (Self::Output, Self::Computation<'_>) {
        (self.feed_forward(&input, |_, _| {}), |output_d| {
            assert_eq!(output_d.shape(), &self.shape);
            (input, [])
        })
        // let mut derivatives = Tensor::<_, B, _>::new(T::zero(), self.shape.clone());
        // (
        //     self.feed_forward(&input, |i, t| derivatives[i] = t.into()),
        //     move |output_d| (self.shape.clone().into_tensor(|i| (output_d[i] * derivatives[i].into()).into()), []),
        // )
    }

    fn update(&mut self, []: &Self::Internal) {}
}

builder::builder! {
    pub struct Builder<(T), (U), (B), const N: usize> {
        activation: A,
    }
}

impl<T, U, B: BackendProvider, A: Activation<U>, const N: usize> LayerBuilder for Builder<A, T, U, B, usizeContainer<N>> {
    type Layer = Convert<T, U, B, A, N>;

    fn build(self, input_shape: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Self::Layer::new(input_shape, self.activation)
    }
}

// impl<T, U, B: BackendProvider, const N: usize> LayerBuilder for Builder<Uninitialized, T, U, B, usizeContainer<N>> {
//     type Layer = Convert<T, U, B, Identity<U>, N>;
//
//     fn build(self, input_shape: <<Self::Layer as Layer>::InputType as Package>::Shapes) -> Self::Layer {
//         Self::Layer::new(input_shape, Identity::new())
//     }
// }
