use std::marker::PhantomData;

use num_traits::Number;
use void::Void;

use crate::{
    layers::{
        convolution::Padding,
        Layer,
        pooling::{Average, Max},
    },
    tensor::{Shape, Tensor},
};

pub struct Pooling2D<S, T: Number, const N: usize> {
    input_shape: Shape,
    output_shape: Shape,
    pool_size: [usize; N],
    padding: [(usize, usize); N],
    strides: [usize; N],
    dilation: [usize; N],
    _marker: PhantomData<(S, T)>,
}

impl<S, T: Number, const N: usize> Pooling2D<S, T, N> {
    pub(crate) fn new(
        input_shape: Shape,
        pool_size: [usize; N],
        padding: [Padding; N],
        strides: [usize; N],
        dilation: [usize; N],
    ) -> Self {
        let padding = padding.map(Padding::resolve);
        let mut output_shape = [0; N + 1];
        for i in 0..N {
            output_shape[i] = (input_shape[i] + padding[i].0 + padding[i].1 - dilation[i] * (pool_size[i] - 1) - 1) / strides[i] + 1
        }
        output_shape[N] = input_shape[N];
        let output_shape = output_shape.into();
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

    fn iter_through_output<F: FnMut([usize; N + 1])>(&self, mut f: F) {
        let mut oi = [0; N + 1];
        loop {
            f(oi);
            if !self.output_shape.increment_index(&mut oi, 1) {
                break;
            }
        }
    }

    fn iter_through_pool<F: FnMut([usize; N])>(&self, mut f: F) {
        let pool_shape: Shape = self.pool_size.into();
        let mut pi = [0; N];
        loop {
            f(pi);
            if !pool_shape.increment_index(&mut pi, 1) {
                break;
            }
        }
    }
}

impl<T: Number, const N: usize> Pooling2D<Average, T, N> {
    fn feed_forward<F: FnMut([usize; N + 1], T)>(&self, input: &Tensor<T>, mut f: F) -> Tensor<T> {
        let mut output = Tensor::zero(self.output_shape.clone());
        self.iter_through_output(|oi| {
            let mut o = T::zero();
            let mut n = 0;
            self.iter_through_pool(|pi| {
                let mut ii = [0; N + 1];
                for i in 0..N {
                    ii[i] = match (oi[i] * self.dilation[i] + pi[i] * self.strides[i]).checked_sub(self.padding[i].0) {
                        None => return,
                        Some(t) => t,
                    }
                }
                ii[N] = oi[N];
                o += input[ii];
                n += 1;
            });
            let n = T::from_i32(n);
            o /= n;
            f(oi, n);
            output[oi] = o
        });
        output
    }
}

impl<T: Number, const N: usize> Layer for Pooling2D<Average, T, N> {
    const INPUT_DIMENSION: usize = 1;
    type InputType = T;
    const REVERSE_INPUT_DIMENSION: usize = 1;
    type ReverseInputType = T;

    const INTERNAL_DIMENSION: usize = 0;
    type InternalType = Void;

    const OUTPUT_DIMENSION: usize = 1;
    type OutputType = T;
    const REVERSE_OUTPUT_DIMENSION: usize = 1;
    type ReverseOutputType = T;

    type BPComputation<'s> = impl FnOnce(Self::ReverseOutputType) ->
    (
        Self::ReverseInputType,
        Self::InternalType
    )
    where
    Self: 's,
    [(); Self::INPUT_DIMENSION]:,
    [(); Self::REVERSE_INPUT_DIMENSION]:,
    [(); Self::INTERNAL_DIMENSION]:,
    [(); Self::OUTPUT_DIMENSION]:,
    [(); Self::REVERSE_OUTPUT_DIMENSION]: ;

    fn get_input_shapes(&self) -> [&Shape; Self::InputType::LEN] {
        [&self.input_shape]
    }

    fn get_input_shape(&self, n: usize) -> &Shape {
        assert_eq!(n, 0);
        &self.input_shape
    }

    fn get_output_shapes(&self) -> [&Shape; Self::OutputType::LEN] {
        [&self.output_shape]
    }

    fn get_output_shape(&self, n: usize) -> &Shape {
        assert_eq!(n, 0);
        &self.output_shape
    }

    fn feed_forward(&self, [input]: Self::InputType) -> Self::OutputType {
        [self.feed_forward(&input, |_, _| {})]
    }

    fn back_propagate(&self, [input]: Self::InputType) -> (Self::OutputType, Self::BPComputation<'_>) where [(); Self::INPUT_DIMENSION]:, [(); Self::REVERSE_INPUT_DIMENSION]:, [(); Self::INTERNAL_DIMENSION]:, [(); Self::OUTPUT_DIMENSION]:, [(); Self::REVERSE_OUTPUT_DIMENSION]: {
        let mut coverage = Tensor::zero(self.output_shape.clone());
        let output = self.feed_forward(&input, |oi, t| coverage[oi] = t);
        (
            [output],
            move |[output_d]| {
                let mut input_d = Tensor::zero(self.input_shape.clone());
                self.iter_through_output(|oi| {
                    let output_d = output_d[oi] / coverage[oi];
                    self.iter_through_pool(|pi| {
                        let mut ii = [0; N + 1];
                        for i in 0..N {
                            ii[i] = match (oi[i] * self.dilation[i] + pi[i] * self.strides[i]).checked_sub(self.padding[i].0) {
                                None => return,
                                Some(t) => t,
                            }
                        }
                        ii[N] = oi[N];
                        input_d[ii] += output_d;
                    });
                });
                ([input_d], [])
            }
        )
    }

    fn update(&mut self, []: &Self::InternalType) {}
}

impl<T: Number + Ord, const N: usize> Pooling2D<Max, T, N> {
    fn feed_forward_<F: FnMut([usize; N + 1], [usize; N + 1])>(&self, input: &Tensor<T>, mut f: F) -> Tensor<T> {
        let mut output = Tensor::zero(self.output_shape.clone());
        self.iter_through_output(|oi| {
            let mut o = T::zero();
            let mut mi = [0; N];
            self.iter_through_pool(|pi| {
                let mut ii = [0; N + 1];
                for i in 0..N {
                    ii[i] = match (oi[i] * self.dilation[i] + pi[i] * self.strides[i]).checked_sub(self.padding[i].0) {
                        None => return,
                        Some(t) => t,
                    }
                }
                ii[N] = oi[N];
                if input[ii] > o {
                    o = input[ii];
                    mi = ii;
                }
            });
            f(oi, mi);
            output[oi] = o
        });
        output
    }
}

impl<T: Number + Ord, const N: usize> Layer for Pooling2D<Max, T, N> {
    const INPUT_DIMENSION: usize = 1;
    type InputType = T;
    const REVERSE_INPUT_DIMENSION: usize = 1;
    type ReverseInputType = T;

    const INTERNAL_DIMENSION: usize = 0;
    type InternalType = Void;

    const OUTPUT_DIMENSION: usize = 1;
    type OutputType = T;
    const REVERSE_OUTPUT_DIMENSION: usize = 1;
    type ReverseOutputType = T;

    type BPComputation<'s> = impl FnOnce(Self::ReverseOutputType) ->
    (
        Self::ReverseInputType,
        Self::InternalType
    )
    where
    Self: 's,
    [(); Self::INPUT_DIMENSION]:,
    [(); Self::REVERSE_INPUT_DIMENSION]:,
    [(); Self::INTERNAL_DIMENSION]:,
    [(); Self::OUTPUT_DIMENSION]:,
    [(); Self::REVERSE_OUTPUT_DIMENSION]: ;

    fn get_input_shapes(&self) -> [&Shape; Self::InputType::LEN] {
        [&self.input_shape]
    }

    fn get_input_shape(&self, n: usize) -> &Shape {
        assert_eq!(n, 0);
        &self.input_shape
    }

    fn get_output_shapes(&self) -> [&Shape; Self::OutputType::LEN] {
        [&self.output_shape]
    }

    fn get_output_shape(&self, n: usize) -> &Shape {
        assert_eq!(n, 0);
        &self.output_shape
    }

    fn feed_forward(&self, [input]: Self::InputType) -> Self::OutputType {
        [self.feed_forward_(&input, |_, _| {})]
    }

    fn back_propagate(&self, [input]: Self::InputType) -> (Self::OutputType, Self::BPComputation<'_>) where [(); Self::INPUT_DIMENSION]:, [(); Self::REVERSE_INPUT_DIMENSION]:, [(); Self::INTERNAL_DIMENSION]:, [(); Self::OUTPUT_DIMENSION]:, [(); Self::REVERSE_OUTPUT_DIMENSION]: {
        let mut coverage = Tensor::default(self.output_shape.clone());
        let output = self.feed_forward_(&input, |oi, ii| coverage[oi] = ii);
        (
            [output],
            move |[output_d]| {
                let mut input_d = Tensor::zero(self.input_shape.clone());
                self.iter_through_output(|oi| {
                    input_d[coverage[oi]] = output_d[oi];
                });
                ([input_d], [])
            }
        )
    }

    fn update(&mut self, []: &Self::InternalType) {}
}
