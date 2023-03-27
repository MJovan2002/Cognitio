use std::array;
use std::ops::Add;
use num_traits::Number;
use tensor::{BackendProvider, Shape, SliceIndex, Tensor};

use crate::{
    activations::Activation,
    initializers::Initializer,
    constraints::Constraint,
    regularizers::Regularizer,
    layers::{
        Layer,
        convolution::Padding,
        Package,
        ToRef,
    },
};

pub struct Conv<
    T: Number,
    B: BackendProvider,
    A: Activation<T>,
    KR,
    BR,
    AR,
    KC,
    BC,
    const N: usize,
> where [(); N + 1]:, [(); N + 2]: {
    input_shape: Shape<{ N + 1 }>,
    output_shape: Shape<{ N + 1 }>,
    kernel: Tensor<T, B, { N + 2 }>,
    bias: Tensor<T, B, { N + 1 }>,
    activation: A,
    padding: [(usize, usize); N],
    strides: [usize; N],
    dilation: [usize; N],
    kernel_regularizer: KR,
    bias_regularizer: BR,
    activity_regularizer: AR,
    kernel_constraint: KC,
    bias_constraint: BC,
}

impl<
    T: Number + for<'s> Add<&'s T, Output=T>,
    B: BackendProvider,
    A: Activation<T>,
    KR,
    BR,
    AR,
    KC,
    BC,
    const N: usize,
> Conv<T, B, A, KR, BR, AR, KC, BC, N> where [(); N + 1]:, [(); N + 2]: {
    pub(crate) fn new<KI: Initializer<T, B, { N + 2 }>, BI: Initializer<T, B, { N + 1 }>>(
        input_shape: Shape<{ N + 1 }>,
        filters: usize,
        kernel_shape: [usize; N],
        mut kernel_initializer: KI,
        mut bias_initializer: BI,
        activation: A,
        padding: [Padding; N],
        strides: [usize; N],
        dilation: [usize; N],
        kernel_regularizer: KR,
        bias_regularizer: BR,
        activity_regularizer: AR,
        kernel_constraint: KC,
        bias_constraint: BC,
    ) -> Self {
        let padding = padding.map(Padding::resolve);
        let mut output_shape = [0; N + 1];
        for i in 0..N {
            output_shape[i] = (input_shape[i] + padding[i].0 + padding[i].1 - dilation[i] * (kernel_shape[i] - 1) - 1) / strides[i] + 1
        }
        output_shape[N] = filters;
        let output_shape = Shape::new(output_shape);
        let mut kernel_shape = array::from_fn(|i| if i < N { kernel_shape[i] } else { 0 });
        kernel_shape[N] = input_shape[N];
        kernel_shape[N + 1] = filters;
        Self {
            kernel: kernel_initializer.initialize(kernel_shape.into()),
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

    fn iter_through_output<F: FnMut([usize; N + 1])>(&self, f: F) {
        let mut index = [0; N + 1];
        loop {
            f(index);
            if !self.output_shape.increment_index(&mut index, 1) {
                break;
            }
        }
    }

    fn iter_through_kernel<F: FnMut([usize; N + 2])>(&self, mut f: F) {
        let mut index = [0; N + 2];
        loop {
            f(index);
            if !self.kernel.shape().increment_index(&mut index, self.output_shape[N]) {
                break;
            }
        }
    }

    fn feed_forward<F: FnMut([usize; N + 1], T)>(&self, input: &Tensor<T, B, { N + 1 }>, mut f: F) -> Tensor<T, B, { N + 1 }> {
        let mut output = Tensor::new(T::zero(), self.output_shape.clone());
        self.iter_through_output(|oi| {
            let mut o = self.bias[oi];
            self.iter_through_kernel(|ki| {
                let mut ii = [0; N + 1];
                for i in 0..N {
                    ii[i] = match (oi[i] * self.dilation[i] + ki[i] * self.strides[i]).checked_sub(self.padding[i].0) {
                        None => return,
                        Some(t) => t,
                    }
                }
                ii[N] = ki[N];
                o += input[ii] * self.kernel[ki]
            });
            f(oi, self.activation.derive(o));
            o = self.activation.activate(o);
            output[oi] = o
        });
        output
    }
}

impl<
    T: Number + for<'s> Add<&'s T, Output=T>,
    B: BackendProvider,
    A: Activation<T>,
    KR: Regularizer<T, B, { N + 2 }>,
    BR: Regularizer<T, B, { N + 1 }>,
    AR: Regularizer<T, B, { N + 1 }>,
    KC: Constraint<T>,
    BC: Constraint<T>,
    const N: usize,
> Layer for Conv<T, B, A, KR, BR, AR, KC, BC, N> where [(); N + 1]:, [(); N + 2]: {
    type InputType = Tensor<T, B, { N + 1 }>;
    type ReverseInputType = Tensor<T, B, { N + 1 }>;
    type InternalType = (Tensor<T, B, { N + 2 }>, Tensor<T, B, { N + 1 }>);
    type OutputType = Tensor<T, B, { N + 1 }>;
    type ReverseOutputType = Tensor<T, B, { N + 1 }>;

    type BPComputation<'s> = impl FnOnce(
        Self::ReverseOutputType
    ) -> (
        Self::ReverseInputType,
        Self::InternalType
    ) + 's where Self: 's;

    fn input_shapes(&self) -> <<Self::InputType as Package>::Shapes as ToRef>::Ref<'_> {
        &self.input_shape
    }

    fn output_shapes(&self) -> <<Self::OutputType as Package>::Shapes as ToRef>::Ref<'_> {
        &self.output_shape
    }

    fn feed_forward(
        &self,
        input: &Self::InputType,
    ) -> Self::OutputType {
        // self.feed_forward(&input, |_, _| {})
        todo!()
    }

    fn back_propagate(
        &self,
        input: Self::InputType,
    ) -> (
        Self::OutputType,
        Self::BPComputation<'_>,
    ) {
        let mut derivatives = Tensor::<_, B, { N + 1 }>::new(T::zero(), self.output_shape.clone());
        let output = todo!();
        // self.feed_forward(&input, |o, t| derivatives[o] = t);
        let activity_reg = self.activity_regularizer.derive(&output);
        (
            output,
            move |output_d| {
                let mut input_d = Tensor::new(T::zero(), self.input_shape.clone());
                let mut kernel_d = self.kernel_regularizer.derive(&self.kernel);
                let bias_d = self.bias_regularizer.derive(&self.bias) + derivatives.slice(SliceIndex::full()).unwrap();
                self.iter_through_output(|oi: [usize; N + 1]| {
                    derivatives[oi] *= output_d[oi] + activity_reg[oi];
                    self.iter_through_kernel(|ki: [usize; N + 2]| {
                        let mut ii = [0; N + 1];
                        for i in 0..N {
                            ii[i] = match (oi[i] * self.dilation[i] + ki[i] * self.strides[i]).checked_sub(self.padding[i].0) {
                                None => return,
                                Some(t) => t,
                            }
                        }
                        ii[N] = ki[N];
                        kernel_d[ki] += derivatives[oi] * input[ii];
                        input_d[ii] += derivatives[oi] * kernel_d[ki];
                    })
                });
                (input_d, (kernel_d, bias_d))
            },
        )
    }

    fn update(&mut self, (kernel, bias): &Self::InternalType) {
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

// impl<
//     const N: usize,
//     T: Number,
//     B: BackendProvider,
//     A: Activation<T>,
//     P: IntoPadding<{ N + 1 }>,
//     S: IntoStride<{ N + 1 }>,
//     D: IntoDilation<{ N + 1 }>,
//     G: IntoGroups,
//     KI: IntoInitializer<T, B, { N + 2 }>,
//     BI: IntoInitializer<T, B, { N + 1 }>,
//     KR: IntoRegularizer<T, B, { N + 2 }>,
//     BR: IntoRegularizer<T, B, { N + 1 }>,
//     AR: IntoRegularizer<T, B, { N + 1 }>,
//     KC: IntoConstraint<T>,
//     BC: IntoConstraint<T>,
// > LayerBuilder for Builder<[usize; N], usize, A, P, S, D, G, KI, BI, KR, BR, AR, KC, BC, T, B, usizeContainer<N>, DimContainer<{ Dim::Dynamic }>> {
//     type Layer = Conv<T, B, A, KR::Regularizer, BR::Regularizer, AR::Regularizer, KC::Constraint, BC::Constraint, N>;
//
//     fn build(self, input_shape: <<Self::Layer as Layer>::InputType as Package>::Shapes) -> Self::Layer {
//         Self::Layer::new(
//             input_shape,
//             self.filters,
//             self.kernel_shape,
//             self.kernel_initializer.into_initializer(),
//             self.bias_initializer.into_initializer(),
//             self.activation,
//             self.padding.into_padding(),
//             self.strides.into_stride(),
//             self.dilation.into_dilation(),
//             // self.groups.into_groups(),
//             self.kernel_regularizer.into_regularizer(),
//             self.bias_regularizer.into_regularizer(),
//             self.activity_regularizer.into_regularizer(),
//             self.kernel_constraint.into_constraint(),
//             self.bias_constraint.into_constraint(),
//         )
//     }
// }
