use num_traits::Number;
use tensor::{BackendProvider, Shape, Tensor};

use crate::{
    activations::Activation,
    constraints::{
        Constraint,
        IntoConstraint
    },
    initializers::{
        Initializer,
        IntoInitializer
    },
    layers::{
        Layer,
        LayerBuilder,
    },
    regularizers::{
        Regularizer,
        IntoRegularizer
    },
    data::{
        Package,
        Uninitialized,
        FromRef
    }
};

pub struct Dense<
    T: Number,
    B: BackendProvider,
    A: Activation<T>,
    KR,
    BR,
    AR,
    KC,
    BC,
    const N: usize,
    const M: usize,
> {
    kernel: Tensor<T, B, 2>,
    bias: Tensor<T, B, M>,
    input_shape: Shape<N>,
    output_shape: Shape<M>,
    activation: A,
    kernel_regularizer: KR,
    bias_regularizer: BR,
    activity_regularizer: AR,
    kernel_constraint: KC,
    bias_constraint: BC,
}

impl<
    T: Number,
    B: BackendProvider,
    A: Activation<T>,
    KR,
    BR,
    AR,
    KC,
    BC,
    const N: usize,
    const M: usize,
> Dense<T, B, A, KR, BR, AR, KC, BC, N, M>
{
    fn new<KI: Initializer<T, B, 2>, BI: Initializer<T, B, M>>(
        input_shape: Shape<N>,
        output_shape: Shape<M>,
        activation: A,
        mut kernel_initializer: KI,
        mut bias_initializer: BI,
        kernel_regularizer: KR,
        bias_regularizer: BR,
        activity_regularizer: AR,
        kernel_constraint: KC,
        bias_constraint: BC,
    ) -> Self {
        Self {
            kernel: kernel_initializer.initialize([
                input_shape.capacity(),
                output_shape.capacity(),
            ].into()),
            bias: bias_initializer.initialize(output_shape.clone()),
            input_shape,
            output_shape,
            activation,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
        }
    }

    fn feed_forward<F: FnMut(usize, T)>(&self, input: &Tensor<T, B, N>, mut f: F) -> Tensor<T, B, M> {
        assert_eq!(input.shape(), &self.input_shape);

        let mut output = Tensor::new(T::zero(), self.bias.shape().clone());
        for o in 0..self.kernel.shape()[1] {
            let mut out = self.bias[o];
            for i in 0..self.kernel.shape()[0] {
                out += input[i] * self.kernel[[i, o]];
            }
            output[o] = self.activation.activate(out);
            f(o, out);
        }
        output
    }
}

impl<
    T: Number,
    B: BackendProvider,
    A: Activation<T>,
    KR: Regularizer<T, B, 2>,
    BR: Regularizer<T, B, M>,
    AR: Regularizer<T, B, M>,
    KC: Constraint<T>,
    BC: Constraint<T>,
    const N: usize,
    const M: usize,
> Layer for Dense<T, B, A, KR, BR, AR, KC, BC, N, M> {
    type Input = Tensor<T, B, N>;
    type ReverseInput = Tensor<T, B, N>;
    type Internal = (Tensor<T, B, 2>, Tensor<T, B, M>);
    type Output = Tensor<T, B, M>;
    type ReverseOutput = Tensor<T, B, M>;

    type Computation<'s> = impl FnOnce(
        Self::ReverseOutput
    ) -> (
        Self::ReverseInput,
        Self::Internal,
    ) + 's where Self: 's;

    fn input_shapes(&self) -> <<Self::Input as Package>::Shapes as FromRef>::Ref<'_> {
        &self.input_shape
    }

    fn output_shapes(&self) -> <<Self::Output as Package>::Shapes as FromRef>::Ref<'_> {
        &self.output_shape
    }

    fn feed_forward(
        &self,
        input: Self::Input,
    ) -> Self::Output {
        self.feed_forward(&input, |_, _| {})
    }

    fn back_propagate(
        &self,
        input: Self::Input,
    ) -> (
        Self::Output,
        Self::Computation<'_>,
    ) {
        let mut derivatives = Tensor::<T, B, _>::new(T::zero(), self.output_shape.clone());
        let output = self.feed_forward(&input, |i, t| derivatives[i] = self.activation.derive(t));
        let activation_reg = self.activity_regularizer.derive(&output);
        (
            output,
            move |output_d| {
                assert_eq!(output_d.shape(), &self.output_shape);
                derivatives
                    .iter_mut()
                    .zip(output_d.iter().zip(activation_reg.iter()))
                    .for_each(|(d, (od, ar))| *d *= *od + *ar);
                let mut input_d = Tensor::new(T::zero(), self.input_shape.clone());
                let mut kernel_d = self.kernel_regularizer.derive(&self.kernel);
                let bias_d = self.bias_regularizer.derive(&self.bias);// todo: + derivatives;
                for i in 0..self.kernel.shape()[0] {
                    let mut id = T::zero();
                    for o in 0..self.kernel.shape()[1] {
                        kernel_d[[i, o]] = derivatives[o] * input[i];
                        id += derivatives[o] * self.kernel[[i, o]];
                    }
                    input_d[i] = id;
                }
                (input_d, (kernel_d, bias_d))
            },
        )
    }

    fn update(&mut self, (kernel, bias): &Self::Internal) {
        self.kernel
            .iter_mut()
            .zip(kernel.iter().copied())
            .for_each(|(a, b)| *a = self.kernel_constraint.constrain(*a - b));
        self.bias
            .iter_mut()
            .zip(bias.iter().copied())
            .for_each(|(a, b)| *a = self.bias_constraint.constrain(*a - b));
    }
}

builder::builder! {
    pub struct Builder<(T), (B), const N: usize, const M: usize> {
        output_shape: SHAPE,
        activation: A,
        kernel_initializer: KI,
        bias_initializer: BI,
        kernel_regularizer: KR,
        bias_regularizer: BR,
        activity_regularizer: AR,
        kernel_constraint: KC,
        bias_constraint: BC,
    }
}

impl<
    T: Number,
    B: BackendProvider,
    A: Activation<T>,
    KI: IntoInitializer<T, B, 2>,
    BI: IntoInitializer<T, B, M>,
    KR: IntoRegularizer<T, B, 2>,
    BR: IntoRegularizer<T, B, M>,
    AR: IntoRegularizer<T, B, M>,
    KC: IntoConstraint<T>,
    BC: IntoConstraint<T>,
    const N: usize,
    const M: usize,
> LayerBuilder for Builder<Shape<M>, A, KI, BI, KR, BR, AR, KC, BC, T, B, usizeContainer<N>, usizeContainer<M>> {
    type Layer = Dense<T, B, A, KR::Regularizer, BR::Regularizer, AR::Regularizer, KC::Constraint, BC::Constraint, N, M>;

    fn build(self, input_shape: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.output_shape,
            self.activation,
            self.kernel_initializer.into_initializer(),
            self.bias_initializer.into_initializer(),
            self.kernel_regularizer.into_regularizer(),
            self.bias_regularizer.into_regularizer(),
            self.activity_regularizer.into_regularizer(),
            self.kernel_constraint.into_constraint(),
            self.bias_constraint.into_constraint(),
        )
    }
}
