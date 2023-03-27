use num_traits::Number;
use tensor::BackendProvider;

use crate::{
    activations::Activation,
    constraints::IntoConstraint,
    data::Uninitialized,
    initializers::IntoInitializer,
    layers::{
        convolution::{Dim, IntoDilation, IntoPadding, IntoStride},
        deconvolution::{
            // deconv::Deconv,
            deconv1d::Deconv1D,
            deconv2d::Deconv2D,
            deconv3d::Deconv3D,
        },
        Layer,
        LayerBuilder,
    },
    regularizers::IntoRegularizer,
};
use crate::data::Package;

mod deconv;
mod deconv1d;
mod deconv2d;
mod deconv3d;

builder::builder! {
    pub struct Builder<(T), (B), const N: usize, const DT: Dim> {
        kernel_shape: SHAPE,
        filters: F,
        activation: A,
        padding: P,
        strides: S,
        dilation: D,
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
    B: BackendProvider<Backend<T>: Clone>,
    A: Activation<T>,
    P: IntoPadding<1>,
    S: IntoStride<1>,
    D: IntoDilation<1>,
    KI: IntoInitializer<T, B, 3>,
    BI: IntoInitializer<T, B, 2>,
    KR: IntoRegularizer<T, B, 3>,
    BR: IntoRegularizer<T, B, 2>,
    AR: IntoRegularizer<T, B, 2>,
    KC: IntoConstraint<T>,
    BC: IntoConstraint<T>,
> LayerBuilder for Builder<[usize; 1], usize, A, P, S, D, KI, BI, KR, BR, AR, KC, BC, T, B, usizeContainer<1>, DimContainer<{ Dim::Static }>> {
    type Layer = Deconv1D<T, B, A, KR::Regularizer, BR::Regularizer, AR::Regularizer, KC::Constraint, BC::Constraint>;

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
            self.kernel_regularizer.into_regularizer(),
            self.bias_regularizer.into_regularizer(),
            self.activity_regularizer.into_regularizer(),
            self.kernel_constraint.into_constraint(),
            self.bias_constraint.into_constraint(),
        )
    }
}

impl<
    T: Number,
    B: BackendProvider<Backend<T>: Clone>,
    A: Activation<T>,
    P: IntoPadding<2>,
    S: IntoStride<2>,
    D: IntoDilation<2>,
    KI: IntoInitializer<T, B, 4>,
    BI: IntoInitializer<T, B, 3>,
    KR: IntoRegularizer<T, B, 4>,
    BR: IntoRegularizer<T, B, 3>,
    AR: IntoRegularizer<T, B, 3>,
    KC: IntoConstraint<T>,
    BC: IntoConstraint<T>,
> LayerBuilder for Builder<[usize; 2], usize, A, P, S, D, KI, BI, KR, BR, AR, KC, BC, T, B, usizeContainer<2>, DimContainer<{ Dim::Static }>> {
    type Layer = Deconv2D<T, B, A, KR::Regularizer, BR::Regularizer, AR::Regularizer, KC::Constraint, BC::Constraint>;

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
            self.kernel_regularizer.into_regularizer(),
            self.bias_regularizer.into_regularizer(),
            self.activity_regularizer.into_regularizer(),
            self.kernel_constraint.into_constraint(),
            self.bias_constraint.into_constraint(),
        )
    }
}

impl<
    T: Number,
    B: BackendProvider<Backend<T>: Clone>,
    A: Activation<T>,
    P: IntoPadding<3>,
    S: IntoStride<3>,
    D: IntoDilation<3>,
    KI: IntoInitializer<T, B, 5>,
    BI: IntoInitializer<T, B, 4>,
    KR: IntoRegularizer<T, B, 5>,
    BR: IntoRegularizer<T, B, 4>,
    AR: IntoRegularizer<T, B, 4>,
    KC: IntoConstraint<T>,
    BC: IntoConstraint<T>,
> LayerBuilder for Builder<[usize; 3], usize, A, P, S, D, KI, BI, KR, BR, AR, KC, BC, T, B, usizeContainer<3>, DimContainer<{ Dim::Static }>> {
    type Layer = Deconv3D<T, B, A, KR::Regularizer, BR::Regularizer, AR::Regularizer, KC::Constraint, BC::Constraint>;

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
            self.kernel_regularizer.into_regularizer(),
            self.bias_regularizer.into_regularizer(),
            self.activity_regularizer.into_regularizer(),
            self.kernel_constraint.into_constraint(),
            self.bias_constraint.into_constraint(),
        )
    }
}

// impl<
//     const N: usize,
//     T: Number,
//     B: BackendProvider<Backend<T>: Clone>,
//     A: Activation<T>,
//     P: IntoPadding<3>,
//     S: IntoStride<3>,
//     D: IntoDilation<3>,
//     KI: IntoInitializer<T, B, { N + 2 }>,
//     BI: IntoInitializer<T, B, { N + 1 }>,
//     KR: IntoRegularizer<T, B, { N + 2 }>,
//     BR: IntoRegularizer<T, B, { N + 1 }>,
//     AR: IntoRegularizer<T, B, { N + 1 }>,
//     KC: IntoConstraint<T>,
//     BC: IntoConstraint<T>,
// > LayerBuilder for Builder<[usize; N], usize, A, P, S, D, KI, BI, KR, BR, AR, KC, BC, T, B, usizeContainer<N>, DimContainer<{ Dim::Dynamic }>> where [(); N + 1]:, [(); N + 2]: {
//     type Layer = Deconv<T, B, A, KR::Regularizer, BR::Regularizer, AR::Regularizer, KC::Constraint, BC::Constraint>;
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
//             self.kernel_regularizer.into_regularizer(),
//             self.bias_regularizer.into_regularizer(),
//             self.activity_regularizer.into_regularizer(),
//             self.kernel_constraint.into_constraint(),
//             self.bias_constraint.into_constraint(),
//         )
//     }
// }
