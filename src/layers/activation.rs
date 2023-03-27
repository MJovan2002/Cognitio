use std::marker::PhantomData;

use num_traits::Number;
use tensor::{Shape, Tensor, VecProvider};
use void::Void;

use crate::{
    activations::Activation as Act,
    layers::{
        Layer,
        LayerBuilder
    },
    data::{Package, FromRef, Uninitialized}
};

pub struct Activation<T, A: Act<T>, const N: usize> {
    activation: A,
    shape: Shape<N>,
    _marker: PhantomData<T>,
}

impl<T: Copy, A: Act<T>, const N: usize> Activation<T, A, N> {
    pub fn new(activation: A, shape: Shape<N>) -> Self {
        Self {
            activation,
            shape,
            _marker: PhantomData,
        }
    }

    fn feed_forward<F: FnMut(usize, T)>(&self, input: Tensor<T, VecProvider, N>, mut f: F) -> Tensor<T, VecProvider, N> {
        assert_eq!(input.shape(), &self.shape);

        let mut o = input;
        o.iter_mut().enumerate().for_each(|(i, t)| {
            f(i, self.activation.derive(*t));
            *t = self.activation.activate(*t);
        });
        o
    }
}

impl<T: Number, A: Act<T>, const N: usize> Layer for Activation<T, A, N> {
    type Input = Tensor<T, VecProvider, N>;
    type ReverseInput = Tensor<T, VecProvider, N>;
    type Internal = [Tensor<Void, VecProvider, 0>; 0];
    type Output = Tensor<T, VecProvider, N>;
    type ReverseOutput = Tensor<T, VecProvider, N>;

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

    fn feed_forward(
        &self,
        input: Self::Input,
    ) -> Self::Output {
        self.feed_forward(input, |_, _| {})
    }

    fn back_propagate(
        &self,
        input: Self::Input,
    ) -> (
        Self::Output,
        Self::Computation<'_>,
    ) {
        let mut derivatives = Tensor::<_, VecProvider, _>::new(T::zero(), self.shape.clone());
        (
            self.feed_forward(input, |i, t| derivatives[i] = t),
            |output_d| {
                assert_eq!(output_d.shape(), &self.shape);
                derivatives
                    .iter_mut()
                    .zip(output_d.iter())
                    .for_each(|(a, b)| *a *= *b);
                (derivatives, [])
            }
        )
    }

    fn update(&mut self, []: &Self::Internal) {}
}

builder::builder! {
    pub struct Builder<const N: usize, (T)> {
        activation: A,
    }
}

impl<T: Number, A: Act<T>, const N: usize> LayerBuilder for Builder<A, usizeContainer<N>, T> {
    type Layer = Activation<T, A, N>;

    fn build(self, input_shapes: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Self::Layer::new(self.activation, input_shapes.clone())
    }
}
