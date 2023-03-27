use num_traits::Number;
use tensor::{BackendProvider, Shape, Tensor};
use void::Void;

use crate::{
    constraints::{
        Constraint,
        IntoConstraint,
    },
    initializers::{
        Initializer,
        IntoInitializer,
    },
    layers::{
        Layer,
        LayerBuilder,
    },
    regularizers::{
        IntoRegularizer,
        Regularizer,
    },
    data::{Package, FromRef, Uninitialized}
};

pub struct Embedding<T, B: BackendProvider, R, C> {
    output_shape: Shape<1>,
    embedding: Tensor<T, B, 2>,
    regularizer: R,
    constraint: C,
}

impl<T: Number, B: BackendProvider, R, C> Embedding<T, B, R, C> {
    pub fn new<I: Initializer<T, B, 2>>(max_size: usize, output_shape: usize, mut initializer: I, regularizer: R, constraint: C) -> Self {
        Self {
            output_shape: [output_shape].into(),
            embedding: initializer.initialize([max_size, output_shape].into()),
            regularizer,
            constraint,
        }
    }
}

impl<T: Number, B: BackendProvider, R: Regularizer<T, B, 1>, C: Constraint<T>> Layer for Embedding<T, B, R, C> {
    type Input = Tensor<usize, B, 0>;
    type ReverseInput = [Void; 0];
    type Internal = Tensor<T, B, 2>;
    type Output = Tensor<T, B, 1>;
    type ReverseOutput = Tensor<T, B, 1>;

    type Computation<'s> = impl FnOnce(
        Self::ReverseOutput
    ) -> (
        Self::ReverseInput,
        Self::Internal,
    ) + 's where Self: 's;

    fn input_shapes(&self) -> <<Self::Input as Package>::Shapes as FromRef>::Ref<'_> {
        static ZERO_SHAPE: Shape<0> = Shape::zero();
        &ZERO_SHAPE
    }

    fn output_shapes(&self) -> <<Self::Output as Package>::Shapes as FromRef>::Ref<'_> {
        &self.output_shape
    }

    fn feed_forward(
        &self,
        input: Self::Input,
    ) -> Self::Output {
        let index = *input;
        self.output_shape.clone().into_tensor(|i| self.embedding[[index, i]])
    }

    fn back_propagate(
        &self,
        input: Self::Input,
    ) -> (
        Self::Output,
        Self::Computation<'_>,
    ) {
        let index = *input;
        let output = self.feed_forward(input);
        let regularization = self.regularizer.derive(&output);
        (
            output,
            move |output_d| {
                assert_eq!(output_d.shape(), &self.output_shape);
                let l = self.output_shape[1];
                let mut internal_d = Tensor::new(T::zero(), self.embedding.shape().clone());
                for i in 0..l {
                    internal_d[[index, i]] = output_d[i] + regularization[i]
                }
                ([], internal_d)
            }
        )
    }

    fn update(&mut self, update: &Self::Internal) {
        self.embedding
            .iter_mut()
            .zip(update.iter().copied())
            .for_each(|(a, b)| *a = self.constraint.constrain(*a - b));
    }
}

builder::builder! {
    pub struct Builder<(T), (B)> {
        output_size: S,
        initializer: I,
        regularizer: R,
        constraint: C,
    }
}

impl<
    T: Number,
    B: BackendProvider,
    I: IntoInitializer<T, B, 2>,
    R: IntoRegularizer<T, B, 1>,
    C: IntoConstraint<T>,
> LayerBuilder for Builder<usize, I, R, C, T, B> {
    type Layer = Embedding<T, B, R::Regularizer, C::Constraint>;

    fn build(self, input_shape: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Self::Layer::new(
            input_shape[0],
            self.output_size,
            self.initializer.into_initializer(),
            self.regularizer.into_regularizer(),
            self.constraint.into_constraint(),
        )
    }
}
