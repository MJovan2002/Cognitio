//! [`Layer`] and [`LayerBuilder`] trait definition

use crate::data::{Package, FromRef};

pub mod add;
pub mod activation;
pub mod convert;
pub mod convolution;
pub mod deconvolution;
pub mod dot;
pub mod dense;
pub mod embedding;
pub mod pooling;
pub mod reshape;
pub mod softmax;
pub mod split;
// todo: add layers

/// [`Layer`]-s are the main building block of the framework.
/// They take one [`Package`] and transform it into another.
pub trait Layer {
    type Input: Package;
    type ReverseInput;
    type Internal;
    type Output: Package;
    type ReverseOutput;

    type Computation<'s>: FnOnce(
        Self::ReverseOutput
    ) -> (
        Self::ReverseInput,
        Self::Internal
    ) + 's where Self: 's;

    fn input_shapes(&self) -> <<Self::Input as Package>::Shapes as FromRef>::Ref<'_>;

    fn output_shapes(&self) -> <<Self::Output as Package>::Shapes as FromRef>::Ref<'_>;

    fn feed_forward(
        &self,
        input: Self::Input,
    ) -> Self::Output;

    fn back_propagate(
        &self,
        input: Self::Input,
    ) -> (
        Self::Output,
        Self::Computation<'_>,
    );

    fn update(&mut self, update: &Self::Internal);
}

/// Builder trait used when creating a [`Model`]
///
/// [`Model`]: crate::model::Model
pub trait LayerBuilder {
    type Layer: Layer;

    fn build(self, input_shape: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer;
}
