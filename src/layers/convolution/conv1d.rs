use std::ops::Add;
use num_traits::Number;
use tensor::{BackendProvider, Shape, SliceIndex, Tensor};

use crate::{
    activations::Activation,
    constraints::Constraint,
    initializers::Initializer,
    regularizers::Regularizer,
    data::{Package, FromRef},
    layers::{
        convolution::*,
        Layer,
    },
};

pub struct Conv1D<
    T: Number,
    B: BackendProvider,
    A: Activation<T>,
    KR,
    BR,
    AR,
    KC,
    BC,
> {
    input_shape: Shape<2>,
    output_shape: Shape<2>,
    kernel: Tensor<T, B, 3>,
    bias: Tensor<T, B, 2>,
    activation: A,
    padding: [(usize, usize); 1],
    strides: [usize; 1],
    dilation: [usize; 1],
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
> Conv1D<T, B, A, KR, BR, AR, KC, BC> {
    pub(crate) fn new<KI: Initializer<T, B, 3>, BI: Initializer<T, B, 2>>(
        input_shape: Shape<2>,
        filters: usize,
        kernel_shape: [usize; 1],
        mut kernel_initializer: KI,
        mut bias_initializer: BI,
        activation: A,
        padding: [Padding; 1],
        strides: [usize; 1],
        dilation: [usize; 1],
        groups: usize,
        kernel_regularizer: KR,
        bias_regularizer: BR,
        activity_regularizer: AR,
        kernel_constraint: KC,
        bias_constraint: BC,
    ) -> Self {
        assert_eq!(filters % groups, 0);
        assert_eq!(input_shape[1] % groups, 0);
        let padding = padding.map(Padding::resolve);
        let output_shape = Shape::new([
            (input_shape[0] + padding[0].0 + padding[0].1 - dilation[0] * (kernel_shape[0] - 1) - 1) / strides[0] + 1,
            filters,
        ]);
        Self {
            kernel: kernel_initializer.initialize([kernel_shape[0], input_shape[1] / groups, filters].into()),
            bias: bias_initializer.initialize(output_shape.clone()),
            input_shape,
            output_shape,
            activation,
            padding,
            strides,
            dilation,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
        }
    }

    fn iter_through_output<F: FnMut([usize; 2])>(&self, mut f: F) {
        for o0 in 0..self.output_shape[0] {
            for o1 in 0..self.output_shape[1] {
                f([o0, o1])
            }
        }
    }

    fn iter_through_kernel<F: FnMut([usize; 2])>(&self, mut f: F) {
        for k0 in 0..self.kernel.shape()[0] {
            for k1 in 0..self.kernel.shape()[1] {
                f([k0, k1])
            }
        }
    }

    fn feed_forward<F: FnMut([usize; 2], T)>(&self, input: &Tensor<T, B, 2>, mut f: F) -> Tensor<T, B, 2> {
        assert_eq!(input.shape(), &self.input_shape);

        let mut output = Tensor::new(T::zero(), self.output_shape.clone());
        self.iter_through_output(|[o0, o1]| {
            let mut o = self.bias[[o0, o1]];
            self.iter_through_kernel(|[k0, k1]| {
                let i0 = (o0 * self.dilation[0] + k0 * self.strides[0]).checked_sub(self.padding[0].0);
                if let Some(i0) = i0 {
                    let groups = self.input_shape[1] / self.kernel.shape()[1];
                    for g in (0..groups).map(|g| g * self.kernel.shape()[1]) {
                        o += input[[i0, g + k1]] * self.kernel[[k0, k1, o1]]
                    }
                }
            });
            f([o0, o1], self.activation.derive(o));
            o = self.activation.activate(o);
            output[[o0, o1]] = o
        });
        output
    }
}

impl<
    T: Number + for<'s> Add<&'s T, Output=T>,
    B: BackendProvider,
    A: Activation<T>,
    KR: Regularizer<T, B, 3>,
    BR: Regularizer<T, B, 2>,
    AR: Regularizer<T, B, 2>,
    KC: Constraint<T>,
    BC: Constraint<T>,
> Layer for Conv1D<T, B, A, KR, BR, AR, KC, BC> {
    type Input = Tensor<T, B, 2>;
    type ReverseInput = Tensor<T, B, 2>;
    type Internal = (Tensor<T, B, 3>, Tensor<T, B, 2>);
    type Output = Tensor<T, B, 2>;
    type ReverseOutput = Tensor<T, B, 2>;

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
        let mut derivatives = Tensor::<T, B, 2>::new(T::zero(), self.output_shape.clone());
        let output = self.feed_forward(&input, |o, t| derivatives[o] = t);
        let activation_reg = self.activity_regularizer.derive(&output);
        (
            output,
            move |output_d| {
                assert_eq!(output_d.shape(), &self.output_shape);
                let mut input_d = Tensor::new(T::zero(), self.input_shape.clone());
                let mut kernel_d = self.kernel_regularizer.derive(&self.kernel);
                let bias_d = self.bias_regularizer.derive(&self.bias) + derivatives.slice(SliceIndex::full()).unwrap();
                self.iter_through_output(|[o0, o1]| {
                    derivatives[[o0, o1]] *= output_d[[o0, o1]] + activation_reg[[o0, o1]];
                    self.iter_through_kernel(|[k0, k1]| {
                        let i0 = (o0 * self.dilation[0] + k0 * self.strides[0]).checked_sub(self.padding[0].0);
                        if let Some(i0) = i0 {
                            let groups = self.input_shape[1] / self.kernel.shape()[1];
                            for g in (0..groups).map(|g| g * self.kernel.shape()[1]) {
                                kernel_d[[k0, k1, o1]] += input[[i0, g + k1]] * derivatives[[o0, o1]];
                                input_d[[i0, g + k1]] += kernel_d[[k0, k1, o1]] * derivatives[[o0, o1]];
                            }
                        }
                    })
                });
                (input_d, (kernel_d, bias_d))
            }
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

impl<
    T: Number + for<'s> Add<&'s T, Output=T>,
    B: BackendProvider,
    A: Activation<T>,
    P: IntoPadding<1>,
    S: IntoStride<1>,
    D: IntoDilation<1>,
    G: IntoGroups,
    KI: IntoInitializer<T, B, 3>,
    BI: IntoInitializer<T, B, 2>,
    KR: IntoRegularizer<T, B, 3>,
    BR: IntoRegularizer<T, B, 2>,
    AR: IntoRegularizer<T, B, 2>,
    KC: IntoConstraint<T>,
    BC: IntoConstraint<T>,
> LayerBuilder for Builder<[usize; 1], usize, A, P, S, D, G, KI, BI, KR, BR, AR, KC, BC, T, B, usizeContainer<1>, DimContainer<{ Dim::Static }>> {
    type Layer = Conv1D<T, B, A, KR::Regularizer, BR::Regularizer, AR::Regularizer, KC::Constraint, BC::Constraint>;

    fn build(self, input_shape: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.filters,
            self.kernel_shape,
            self.kernel_initializer.into_initializer(),
            self.bias_initializer.into_initializer(),
            self.activation,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
            self.groups.into_groups(),
            self.kernel_regularizer.into_regularizer(),
            self.bias_regularizer.into_regularizer(),
            self.activity_regularizer.into_regularizer(),
            self.kernel_constraint.into_constraint(),
            self.bias_constraint.into_constraint(),
        )
    }
}
