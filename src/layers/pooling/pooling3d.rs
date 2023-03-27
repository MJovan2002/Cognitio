use std::marker::PhantomData;

use num_traits::Number;
use tensor::{BackendProvider, Shape, Tensor};
use void::Void;

use crate::{
    data::{Package, FromRef},
    layers::{
        convolution::{IntoDilation, IntoPadding, IntoStride, Padding},
        Layer,
        pooling::*,
    },
};

pub struct Pooling3D<const S: PoolingType, T: Number, B> {
    input_shape: Shape<4>,
    output_shape: Shape<4>,
    pool_size: [usize; 3],
    padding: [(usize, usize); 3],
    strides: [usize; 3],
    dilation: [usize; 3],
    _marker: PhantomData<(T, B)>,
}

impl<const S: PoolingType, T: Number, B> Pooling3D<S, T, B> {
    pub(crate) fn new(
        input_shape: Shape<4>,
        pool_size: [usize; 3],
        padding: [Padding; 3],
        strides: [usize; 3],
        dilation: [usize; 3],
    ) -> Self {
        let padding = padding.map(Padding::resolve);
        let output_shape = [
            (input_shape[0] + padding[0].0 + padding[0].1 - dilation[0] * (pool_size[0] - 1) - 1) / strides[0] + 1,
            (input_shape[1] + padding[1].0 + padding[1].1 - dilation[1] * (pool_size[1] - 1) - 1) / strides[1] + 1,
            (input_shape[2] + padding[2].0 + padding[2].1 - dilation[2] * (pool_size[2] - 1) - 1) / strides[2] + 1,
            input_shape[3],
        ].into();
        Self {
            input_shape,
            output_shape,
            pool_size,
            padding,
            strides,
            dilation,
            _marker: PhantomData,
        }
    }

    fn iter_through_output<F: FnMut([usize; 4])>(&self, mut f: F) {
        for o0 in 0..self.output_shape[0] {
            for o1 in 0..self.output_shape[1] {
                for o2 in 0..self.output_shape[2] {
                    for o3 in 0..self.output_shape[3] {
                        f([o0, o1, o2, o3])
                    }
                }
            }
        }
    }

    fn iter_through_pool<F: FnMut([usize; 3])>(&self, mut f: F) {
        for p0 in 0..self.pool_size[0] {
            for p1 in 0..self.pool_size[1] {
                for p2 in 0..self.pool_size[2] {
                    f([p0, p1, p2])
                }
            }
        }
    }
}

impl<T: Number + From<i32>, B: BackendProvider> Pooling3D<{ PoolingType::Average }, T, B> {
    fn feed_forward<F: FnMut([usize; 4], T)>(&self, input: &Tensor<T, B, 4>, mut f: F) -> Tensor<T, B, 4> {
        assert_eq!(input.shape(), &self.input_shape);

        let mut output = Tensor::new(T::zero(), self.output_shape.clone());
        self.iter_through_output(|[o0, o1, o2, o3]| {
            let mut o = T::zero();
            let mut n = 0;
            self.iter_through_pool(|[p0, p1, p2]| {
                let i0 = (o0 * self.dilation[0] + p0 * self.strides[0]).checked_sub(self.padding[0].0);
                let i1 = (o1 * self.dilation[1] + p1 * self.strides[1]).checked_sub(self.padding[1].0);
                let i2 = (o2 * self.dilation[2] + p2 * self.strides[2]).checked_sub(self.padding[2].0);
                if let (Some(i0), Some(i1), Some(i2)) = (i0, i1, i2) {
                    o += input[[i0, i1, i2, o3]];
                    n += 1;
                }
            });
            let n = T::from(n);
            o /= n;
            f([o0, o1, o2, o3], n);
            output[[o0, o1, o2, o3]] = o
        });
        output
    }
}

impl<T: Number + From<i32>, B: BackendProvider> Layer for Pooling3D<{ PoolingType::Average }, T, B> {
    type Input = Tensor<T, B, 4>;
    type ReverseInput = Tensor<T, B, 4>;
    type Internal = [Void; 0];
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

    fn feed_forward(&self, input: Self::Input) -> Self::Output {
        self.feed_forward(&input, |_, _| {})
    }

    fn back_propagate(&self, input: Self::Input) -> (Self::Output, Self::Computation<'_>) {
        // let mut coverage = Tensor::new(T::zero(), self.output_shape.clone());
        // let output = self.feed_forward(&input, |oi, t| coverage[oi] = t);
        let output = self.feed_forward(&input, |_, _| {});
        (
            output,
            move |output_d| {
                todo!()
                // assert_eq!(output_d.shape(), &self.output_shape);
                // let mut input_d = Tensor::new(T::zero(), self.input_shape.clone());
                // self.iter_through_output(|[o0, o1, o2, o3]| {
                //     let output_d = output_d[[o0, o1, o2, o3]] / coverage[[o0, o1, o2, o3]];
                //     self.iter_through_pool(|[p0, p1, p2]| {
                //         let i0 = (o0 * self.dilation[0] + p0 * self.strides[0]).checked_sub(self.padding[0].0);
                //         let i1 = (o1 * self.dilation[1] + p1 * self.strides[1]).checked_sub(self.padding[1].0);
                //         let i2 = (o2 * self.dilation[2] + p2 * self.strides[2]).checked_sub(self.padding[2].0);
                //         if let (Some(i0), Some(i1), Some(i2)) = (i0, i1, i2) {
                //             input_d[[i0, i1, i2, o3]] += output_d;
                //         }
                //     });
                // });
                // (input_d, [])
            }
        )
    }

    fn update(&mut self, []: &Self::Internal) {}
}

impl<T: Number + Ord, B: BackendProvider> Pooling3D<{ PoolingType::Max }, T, B> {
    fn feed_forward_<F: FnMut([usize; 4], [usize; 4])>(&self, input: &Tensor<T, B, 4>, mut f: F) -> Tensor<T, B, 4> {
        assert_eq!(input.shape(), &self.input_shape);

        let mut output = Tensor::new(T::zero(), self.output_shape.clone());
        self.iter_through_output(|[o0, o1, o2, o3]| {
            let mut o = T::zero();
            let mut mi = [0, 0, 0, 0];
            self.iter_through_pool(|[p0, p1, p2]| {
                let i0 = (o0 * self.dilation[0] + p0 * self.strides[0]).checked_sub(self.padding[0].0);
                let i1 = (o1 * self.dilation[1] + p1 * self.strides[1]).checked_sub(self.padding[1].0);
                let i2 = (o2 * self.dilation[2] + p2 * self.strides[2]).checked_sub(self.padding[2].0);
                if let (Some(i0), Some(i1), Some(i2)) = (i0, i1, i2) {
                    if input[[i0, i1, i2, o3]] > o {
                        o = input[[i0, i1, i2, o3]];
                        mi = [i0, i1, i2, o3];
                    }
                }
            });
            f([o0, o1, o2, o3], mi);
            output[[o0, o1, o2, o3]] = o
        });
        output
    }
}

impl<T: Number + Ord, B: BackendProvider> Layer for Pooling3D<{ PoolingType::Max }, T, B> {
    type Input = Tensor<T, B, 4>;
    type ReverseInput = Tensor<T, B, 4>;
    type Internal = [Void; 0];
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

    fn feed_forward(&self, input: Self::Input) -> Self::Output {
        self.feed_forward_(&input, |_, _| {})
    }

    fn back_propagate(&self, input: Self::Input) -> (Self::Output, Self::Computation<'_>) {
        // let mut coverage = Tensor::new(T::zero(), self.output_shape.clone());
        // let output = self.feed_forward_(&input, |oi, ii| coverage[oi] = ii);
        let output = self.feed_forward_(&input, |_, _| {});
        (
            output,
            move |output_d| {
                todo!()
                // assert_eq!(output_d.shape(), &self.output_shape);
                // let mut input_d = Tensor::new(T::zero(), self.input_shape.clone());
                // self.iter_through_output(|oi| {
                //     input_d[coverage[oi]] = output_d[oi];
                // });
                // (input_d, [])
            }
        )
    }

    fn update(&mut self, []: &Self::Internal) {}
}

impl<
    T: Number + From<i32>,
    B: BackendProvider,
    P: IntoPadding<3>,
    S: IntoStride<3>,
    D: IntoDilation<3>,
> LayerBuilder for Builder<[usize; 3], P, S, D, T, B, usizeContainer<3>, DimContainer<{ Dim::Static }>, PoolingTypeContainer<{ PoolingType::Average }>> {
    type Layer = Pooling3D<{ PoolingType::Average }, T, B>;

    fn build(self, input_shape: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.pool_shape,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
        )
    }
}

impl<
    T: Number + Ord,
    B: BackendProvider,
    P: IntoPadding<3>,
    S: IntoStride<3>,
    D: IntoDilation<3>,
> LayerBuilder for Builder<[usize; 3], P, S, D, T, B, usizeContainer<3>, DimContainer<{ Dim::Static }>, PoolingTypeContainer<{ PoolingType::Max }>> {
    type Layer = Pooling3D<{ PoolingType::Max }, T, B>;

    fn build(self, input_shape: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.pool_shape,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
        )
    }
}
