use crate::{
    constraints::IntoConstraint,
    data::Uninitialized,
    initializers::IntoInitializer,
    layers::LayerBuilder,
    regularizers::IntoRegularizer,
};

// mod conv;
mod conv1d;
mod conv2d;
mod conv3d;

#[derive(Copy, Clone)]
pub enum Padding {
    None,
    Symmetrical(usize),
    Asymmetrical(usize, usize),
}

impl Padding {
    pub const fn resolve(self) -> (usize, usize) {
        match self {
            Padding::None => (0, 0),
            Padding::Symmetrical(p) => (p, p),
            Padding::Asymmetrical(p0, p1) => (p0, p1),
        }
    }
}

pub trait IntoPadding<const N: usize> {
    fn into_padding(self) -> [Padding; N];
}

impl<const N: usize> IntoPadding<N> for Uninitialized {
    fn into_padding(self) -> [Padding; N] {
        [Padding::None; N]
    }
}

impl<const N: usize> IntoPadding<N> for [Padding; N] {
    fn into_padding(self) -> [Padding; N] {
        self
    }
}

pub trait IntoStride<const N: usize> {
    fn into_stride(self) -> [usize; N];
}

impl<const N: usize> IntoStride<N> for Uninitialized {
    fn into_stride(self) -> [usize; N] {
        [1; N]
    }
}

impl<const N: usize> IntoStride<N> for [usize; N] {
    fn into_stride(self) -> [usize; N] {
        self
    }
}

pub trait IntoDilation<const N: usize> {
    fn into_dilation(self) -> [usize; N];
}

impl<const N: usize> IntoDilation<N> for Uninitialized {
    fn into_dilation(self) -> [usize; N] {
        [1; N]
    }
}

impl<const N: usize> IntoDilation<N> for [usize; N] {
    fn into_dilation(self) -> [usize; N] {
        self
    }
}

pub trait IntoGroups {
    fn into_groups(self) -> usize;
}

impl IntoGroups for Uninitialized {
    fn into_groups(self) -> usize {
        1
    }
}

impl IntoGroups for usize {
    fn into_groups(self) -> usize {
        self
    }
}

#[derive(Eq, PartialEq, Copy, Clone)]
pub enum Dim {
    Static,
    Dynamic,
}

builder::builder! {
    pub struct Builder<(T), (B), const N: usize, const DT: Dim> {
        kernel_shape: SHAPE,
        filters: F,
        activation: A,
        padding: P,
        strides: S,
        dilation: D,
        groups: G,
        kernel_initializer: KI,
        bias_initializer: BI,
        kernel_regularizer: KR,
        bias_regularizer: BR,
        activity_regularizer: AR,
        kernel_constraint: KC,
        bias_constraint: BC,
    }
}
