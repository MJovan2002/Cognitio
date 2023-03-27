#![allow(unused_qualifications)]

use std::marker::Destruct;

use crate::{
    layers::{Layer, LayerBuilder},
    model::{Model, model_tuple::ModelTuple},
    data::{Package, FromRef},
};

pub struct Sequential<M> {
    inner: M,
}

impl<M> Sequential<M> {
    const fn new(inner: M) -> Self {
        Self { inner }
    }
}

impl Sequential<()> {
    pub(crate) const fn empty() -> Self {
        Self::new(())
    }
}

impl<M: ~ const Insertable + ~ const Destruct> Sequential<M> {
    pub const fn add_layer<L>(self, layer: L) -> Sequential<M::Output<L>> {
        Sequential::new(self.inner.insert(layer))
    }
}

impl<
    L: Layer,
    M: Buildable<Input=<<Model<L> as Layer>::Input as Package>::Shapes, Output=L>
> LayerBuilder for Sequential<M> {
    type Layer = Model<L>;

    fn build(self, input_shape: <<Self::Layer as Layer>::Input as Package>::Shapes) -> Self::Layer {
        Model::from_inner(self.inner.build(input_shape))
    }
}

#[const_trait]
pub trait Insertable {
    type Output<T>;

    fn insert<T>(self, t: T) -> Self::Output<T>;
}

impl const Insertable for () {
    type Output<T> = T;

    fn insert<T>(self, t: T) -> Self::Output<T> {
        t
    }
}

auto trait NotUnit {}

impl ! NotUnit for () {}

auto trait NotPair {}

impl<A, B> ! NotPair for (A, B) {}

impl<A: NotPair + NotUnit> const Insertable for A {
    type Output<T> = (A, T);

    fn insert<T>(self, t: T) -> Self::Output<T> {
        (self, t)
    }
}

impl<A: ~ const Destruct, B: ~ const Insertable + ~ const Destruct> const Insertable for (A, B) {
    type Output<T> = (A, B::Output<T>);

    fn insert<T>(self, t: T) -> Self::Output<T> {
        let (a, b) = self;
        (a, b.insert(t))
    }
}

pub trait Buildable {
    type Input;
    type Output;

    fn build(self, input: Self::Input) -> Self::Output;
}

impl<B: LayerBuilder> Buildable for B {
    type Input = <<B::Layer as Layer>::Input as Package>::Shapes;
    type Output = B::Layer;

    fn build(self, input: Self::Input) -> Self::Output {
        self.build(input)
    }
}

impl<
    B: LayerBuilder,
    M: Buildable<
        Input=<<B::Layer as Layer>::Output as Package>::Shapes,
        Output: Layer<
            Input=<B::Layer as Layer>::Output,
            ReverseInput=<B::Layer as Layer>::ReverseOutput
        >
    >
> Buildable for (B, M) {
    type Input = <<B::Layer as Layer>::Input as Package>::Shapes;
    type Output = ModelTuple<
        B::Layer,
        M::Output,
        <B::Layer as Layer>::Input,
        (),
        (),
        <M::Output as Layer>::Output,
        <B::Layer as Layer>::ReverseInput,
        <M::Output as Layer>::ReverseOutput,
        (),
        ()
    >;

    fn build(self, input: Self::Input) -> Self::Output {
        let layer = self.0.build(input);
        let sub_model = self.1.build(FromRef::from_ref(layer.output_shapes()));
        ModelTuple::new(
            layer,
            sub_model,
            |input| ((), input),
            |(), layer_output| ((), layer_output),
            |(), model_output| model_output,
            |model_derivatives| ((), model_derivatives),
            |(), layer_derivatives| (layer_derivatives, ()),
            |input_derivatives, ()| input_derivatives,
        )
    }
}
