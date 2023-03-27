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

pub struct Pooling2D<const S: PoolingType, T: Number, B> {
    input_shape: Shape<3>,
    output_shape: Shape<3>,
    pool_size: [usize; 2],
    padding: [(usize, usize); 2],
    strides: [usize; 2],
    dilation: [usize; 2],
    _marker: PhantomData<(T, B)>,
}

impl<const S: PoolingType, T: Number, B> Pooling2D<S, T, B> {
    pub(crate) fn new(
        input_shape: Shape<3>,
        pool_size: [usize; 2],
        padding: [Padding; 2],
        strides: [usize; 2],
        dilation: [usize; 2],
    ) -> Self {
        let padding = padding.map(Padding::resolve);
        let output_shape = [
            (input_shape[0] + padding[0].0 + padding[0].1 - dilation[0] * (pool_size[0] - 1) - 1) / strides[0] + 1,
            (input_shape[1] + padding[1].0 + padding[1].1 - dilation[1] * (pool_size[1] - 1) - 1) / strides[1] + 1,
            input_shape[2],
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

    fn iter_through_output<F: FnMut([usize; 3])>(&self, mut f: F) {
        for o0 in 0..self.output_shape[0] {
            for o1 in 0..self.output_shape[1] {
                for o2 in 0..self.output_shape[2] {
                    f([o0, o1, o2])
                }
            }
        }
    }

    fn iter_through_pool<F: FnMut([usize; 2])>(&self, mut f: F) {
        for p0 in 0..self.pool_size[0] {
            for p1 in 0..self.pool_size[1] {
                f([p0, p1])
            }
        }
    }
}

impl<T: Number + From<i32>, B: BackendProvider> Pooling2D<{ PoolingType::Average }, T, B> {
    fn feed_forward<F: FnMut([usize; 3], T)>(&self, input: &Tensor<T, B, 3>, mut f: F) -> Tensor<T, B, 3> {
        assert_eq!(input.shape(), &self.input_shape);

        let mut output = Tensor::new(T::zero(), self.output_shape.clone());
        self.iter_through_output(|[o0, o1, o2]| {
            let mut o = T::zero();
            let mut n = 0;
            self.iter_through_pool(|[p0, p1]| {
                let i0 = (o0 * self.dilation[0] + p0 * self.strides[0]).checked_sub(self.padding[0].0);
                let i1 = (o1 * self.dilation[1] + p1 * self.strides[1]).checked_sub(self.padding[1].0);
                if let (Some(i0), Some(i1)) = (i0, i1) {
                    o += input[[i0, i1, o2]];
                    n += 1;
                }
            });
            let n = T::from(n);
            o /= n;
            f([o0, o1, o2], n);
            output[[o0, o1, o2]] = o
        });
        output
    }
}

impl<T: Number + From<i32>, B: BackendProvider> Layer for Pooling2D<{ PoolingType::Average }, T, B> {
    type Input = Tensor<T, B, 3>;
    type ReverseInput = Tensor<T, B, 3>;
    type Internal = [Void; 0];
    type Output = Tensor<T, B, 3>;
    type ReverseOutput = Tensor<T, B, 3>;

    type Computation<'s> = impl FnOnce(
        Self::ReverseOutput) ->
    (
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
                // self.iter_through_output(|[o0, o1, o2]| {
                //     let output_d = output_d[[o0, o1, o2]] / coverage[[o0, o1, o2]];
                //     self.iter_through_pool(|[p0, p1]| {
                //         let i0 = (o0 * self.dilation[0] + p0 * self.strides[0]).checked_sub(self.padding[0].0);
                //         let i1 = (o1 * self.dilation[1] + p1 * self.strides[1]).checked_sub(self.padding[1].0);
                //         if let (Some(i0), Some(i1)) = (i0, i1) {
                //             input_d[[i0, i1, o2]] += output_d;
                //         }
                //     });
                // });
                // (input_d, [])
            }
        )
    }

    fn update(&mut self, []: &Self::Internal) {}
}

impl<T: Number + Ord, B: BackendProvider> Pooling2D<{ PoolingType::Max }, T, B> {
    fn feed_forward_<F: FnMut([usize; 3], [usize; 3])>(&self, input: &Tensor<T, B, 3>, mut f: F) -> Tensor<T, B, 3> {
        assert_eq!(input.shape(), &self.input_shape);

        let mut output = Tensor::new(T::zero(), self.output_shape.clone());
        self.iter_through_output(|[o0, o1, o2]| {
            let mut o = T::zero();
            let mut mi = [0, 0, 0];
            self.iter_through_pool(|[p0, p1]| {
                let i0 = (o0 * self.dilation[0] + p0 * self.strides[0]).checked_sub(self.padding[0].0);
                let i1 = (o1 * self.dilation[1] + p1 * self.strides[1]).checked_sub(self.padding[1].0);
                if let (Some(i0), Some(i1)) = (i0, i1) {
                    if input[[i0, i1, o2]] > o {
                        o = input[[i0, i1, o2]];
                        mi = [i0, i1, o2];
                    }
                }
            });
            f([o0, o1, o2], mi);
            output[[o0, o1, o2]] = o
        });
        output
    }
}

impl<T: Number + Ord, B: BackendProvider> Layer for Pooling2D<{ PoolingType::Max }, T, B> {
    type Input = Tensor<T, B, 3>;
    type ReverseInput = Tensor<T, B, 3>;
    type Internal = [Void; 0];
    type Output = Tensor<T, B, 3>;
    type ReverseOutput = Tensor<T, B, 3>;

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
    P: IntoPadding<2>,
    S: IntoStride<2>,
    D: IntoDilation<2>,
> LayerBuilder for Builder<[usize; 2], P, S, D, T, B, usizeContainer<2>, DimContainer<{ Dim::Static }>, PoolingTypeContainer<{ PoolingType::Average }>> {
    type Layer = Pooling2D<{ PoolingType::Average }, T, B>;

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
    P: IntoPadding<2>,
    S: IntoStride<2>,
    D: IntoDilation<2>,
> LayerBuilder for Builder<[usize; 2], P, S, D, T, B, usizeContainer<2>, DimContainer<{ Dim::Static }>, PoolingTypeContainer<{ PoolingType::Max }>> {
    type Layer = Pooling2D<{ PoolingType::Max }, T, B>;

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
