use crate::{
    layers::{
        convolution::Dim,
        LayerBuilder,
    },
    data::Uninitialized,
};

// pub mod pooling;
pub mod pooling1d;
pub mod pooling2d;
pub mod pooling3d;

#[derive(Eq, PartialEq)]
pub enum PoolingType {
    Average,
    Max,
}

builder::builder! {
    pub struct Builder<(T), (B), const N: usize, const DT: Dim, const M: PoolingType> {
        pool_shape: SHAPE,
        padding: P,
        strides: S,
        dilation: D,
    }
}
