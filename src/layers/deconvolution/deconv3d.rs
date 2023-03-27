use num_traits::Number;
use tensor::{BackendProvider, Shape, Tensor};

use crate::{
    activations::Activation,
    constraints::Constraint,
    initializers::Initializer,
    layers::{
        convolution::Padding,
        Layer,
    },
    regularizers::Regularizer,
};
use crate::data::Package;
use crate::data::FromRef;

pub struct Deconv3D<
    T: Number,
    B: BackendProvider,
    A: Activation<T>,
    KR,
    BR,
    AR,
    KC,
    BC,
> {
    input_shape: Shape<4>,
    output_shape: Shape<4>,
    kernel: Tensor<T, B, 5>,
    bias: Tensor<T, B, 4>,
    activation: A,
    padding: [(usize, usize); 3],
    strides: [usize; 3],
    dilation: [usize; 3],
    kernel_regularizer: KR,
    bias_regularizer: BR,
    activity_regularizer: AR,
    kernel_constraint: KC,
    bias_constraint: BC,
}

impl<
    T: Number,
    B: BackendProvider<Backend<T>: Clone>,
    A: Activation<T>,
    KR,
    BR,
    AR,
    KC,
    BC,
> Deconv3D<T, B, A, KR, BR, AR, KC, BC> {
    pub(crate) fn new<KI: Initializer<T, B, 5>, BI: Initializer<T, B, 4>>(
        input_shape: Shape<4>,
        filters: usize,
        kernel_shape: [usize; 3],
        mut kernel_initializer: KI,
        mut bias_initializer: BI,
        activation: A,
        padding: [Padding; 3],
        strides: [usize; 3],
        dilation: [usize; 3],
        kernel_regularizer: KR,
        bias_regularizer: BR,
        activity_regularizer: AR,
        kernel_constraint: KC,
        bias_constraint: BC,
    ) -> Self {
        let padding = padding.map(Padding::resolve);
        let output_shape = Shape::new([
            (input_shape[0] - 1) * strides[0] + dilation[0] * (kernel_shape[0] - 1) + 1 - padding[0].0 - padding[0].1,
            (input_shape[1] - 1) * strides[1] + dilation[1] * (kernel_shape[1] - 1) + 1 - padding[1].0 - padding[1].1,
            (input_shape[2] - 1) * strides[2] + dilation[2] * (kernel_shape[2] - 1) + 1 - padding[2].0 - padding[2].1,
            filters,
        ]);
        Self {
            kernel: kernel_initializer.initialize([kernel_shape[0], kernel_shape[1], kernel_shape[2], input_shape[3], filters].into()),
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

    fn iter_through_input<F: FnMut([usize; 4])>(&self, mut f: F) {
        for i0 in 0..self.input_shape[0] {
            for i1 in 0..self.input_shape[1] {
                for i2 in 0..self.input_shape[2] {
                    for i3 in 0..self.input_shape[3] {
                        f([i0, i1, i2, i3])
                    }
                }
            }
        }
    }

    fn iter_through_kernel<F: FnMut([usize; 4])>(&self, mut f: F) {
        for k0 in 0..self.kernel.shape()[0] {
            for k1 in 0..self.kernel.shape()[1] {
                for k2 in 0..self.kernel.shape()[2] {
                    for k4 in 0..self.kernel.shape()[4] {
                        f([k0, k1, k2, k4])
                    }
                }
            }
        }
    }

    fn feed_forward<F: FnMut([usize; 4], T)>(&self, input: &Tensor<T, B, 4>, mut f: F) -> Tensor<T, B, 4> {
        assert_eq!(input.shape(), &self.input_shape);

        let mut output = self.bias.clone();
        self.iter_through_input(|[i0, i1, i2, i3]| {
            self.iter_through_kernel(|[k0, k1, k2, k4]| {
                let o0 = (i0 * self.dilation[0] + k0 * self.strides[0]).checked_sub(self.padding[0].0);
                let o1 = (i1 * self.dilation[1] + k1 * self.strides[1]).checked_sub(self.padding[1].0);
                let o2 = (i2 * self.dilation[2] + k2 * self.strides[2]).checked_sub(self.padding[2].0);
                if let (Some(o0), Some(o1), Some(o2)) = (o0, o1, o2) {
                    output[[o0, o1, o2, k4]] += input[[i0, i1, i2, i3]] * self.kernel[[k0, k1, k2, i3, k4]]
                }
            });
        });
        output.iter_mut_indexed().for_each(|(oi, t)| {
            f(oi, *t);
            *t = self.activation.activate(*t)
        });
        output
    }
}

impl<
    T: Number,
    B: BackendProvider<Backend<T>: Clone>,
    A: Activation<T>,
    KR: Regularizer<T, B, 5>,
    BR: Regularizer<T, B, 4>,
    AR: Regularizer<T, B, 4>,
    KC: Constraint<T>,
    BC: Constraint<T>,
> Layer for Deconv3D<T, B, A, KR, BR, AR, KC, BC> {
    type Input = Tensor<T, B, 4>;
    type ReverseInput = Tensor<T, B, 4>;
    type Internal = (Tensor<T, B, 5>, Tensor<T, B, 4>);
    type Output = Tensor<T, B, 4>;
    type ReverseOutput = Tensor<T, B, 4>;

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
        let mut derivatives = Tensor::<_, B, _>::new(T::zero(), self.output_shape.clone());
        let output = self.feed_forward(&input, |o, t| derivatives[o] = t);
        let activation_reg = self.activity_regularizer.derive(&output);
        (
            output,
            move |output_d| {
                assert_eq!(output_d.shape(), &self.output_shape);
                let mut input_d = Tensor::new(T::zero(), self.input_shape.clone());
                let mut kernel_d = self.kernel_regularizer.derive(&self.kernel);
                let bias_d = self.bias_regularizer.derive(&self.bias);// todo: + derivatives;
                derivatives.iter_mut().zip(output_d.iter().copied().zip(activation_reg.iter().copied())).for_each(|(a, (b, c))| *a *= b + c);
                self.iter_through_input(|[i0, i1, i2, i3]| {
                    let mut i = T::zero();
                    self.iter_through_kernel(|[k0, k1, k2, k4]| {
                        let o0 = (i0 * self.dilation[0] + k0 * self.strides[0]).checked_sub(self.padding[0].0);
                        let o1 = (i1 * self.dilation[1] + k1 * self.strides[1]).checked_sub(self.padding[1].0);
                        let o2 = (i2 * self.dilation[2] + k2 * self.strides[2]).checked_sub(self.padding[2].0);
                        if let (Some(o0), Some(o1), Some(o2)) = (o0, o1, o2) {
                            i += self.kernel[[k0, k1, k2, i3, k4]] * derivatives[[o0, o1, o2, k4]];
                            kernel_d[[k0, k1, k2, i3, k4]] += input[[i0, i1, i2, i3]] * derivatives[[o0, o1, o2, k4]];
                        }
                    });
                    input_d[[i0, i1, i2, i3]] = i;
                });

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
