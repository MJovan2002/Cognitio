use crate::{
    layers::Layer,
    data::{Package, FromRef},
    optimizers::Optimizer,
    trainer::Trainer,
};
use sequential::Sequential;

mod sequential;
pub mod model_tuple;

pub struct Model<M> {
    pub model: M,
}

impl<M: Layer> Model<M> {
    pub(crate) const fn from_inner(model: M) -> Self {
        Self { model }
    }

    // pub fn new<I: Inputs, O: Outputs>(inputs: I, outputs: O) -> Self {
    //     todo!()
    // }
}

impl<M: Layer> Model<M> {
    pub fn compile<O: Optimizer<M::Internal>>(&mut self, optimizer: O) -> Trainer<'_, M, O> {
        Trainer::new(self, optimizer)
    }
}

impl Model<()> {
    pub const fn sequential() -> Sequential<()> {
        Sequential::empty()
    }
}

impl<M: Layer> Layer for Model<M> {
    type Input = M::Input;
    type ReverseInput = M::ReverseInput;
    type Internal = M::Internal;
    type Output = M::Output;
    type ReverseOutput = M::ReverseOutput;
    type Computation<'s> = M::Computation<'s> where Self: 's;

    fn input_shapes(&self) -> <<Self::Input as Package>::Shapes as FromRef>::Ref<'_> {
        self.model.input_shapes()
    }

    fn output_shapes(&self) -> <<Self::Output as Package>::Shapes as FromRef>::Ref<'_> {
        self.model.output_shapes()
    }

    fn feed_forward(&self, input: Self::Input) -> Self::Output {
        self.model.feed_forward(input)
    }

    fn back_propagate(&self, input: Self::Input) -> (Self::Output, Self::Computation<'_>) {
        self.model.back_propagate(input)
    }

    fn update(&mut self, update: &Self::Internal) {
        self.model.update(update)
    }
}

// pub struct Input {}
//
// impl Input {
//     pub fn get_link(&mut self) -> Link<'_> {
//         todo!()
//     }
// }
//
// pub trait Inputs {}
//
// pub struct Link<'s> {
//     _marker: PhantomData<&'s ()>,
// }
//
// impl<'s> Link<'s> {
//     pub fn output(self) -> Output {
//         todo!()
//     }
//
//     pub fn output_with_loss<L>(self, loss: L) -> Output {
//         todo!()
//     }
// }
//
// pub trait Links<B: LayerBuilder> {
//     fn get_info(&self) -> <<B::Layer as Layer>::Input as Package>::Shapes;
// }
//
// pub struct Output {}
//
// pub trait Outputs {}
//
// pub fn combine<I, O, B: LayerBuilder, L: Links<B>, M: Layer>(builder: B, links: L) -> () {
//     let t = |sub: M| ModelTuple::new(
//         builder.build(links.get_info()),
//         sub,
//         |input: I| (todo!(), todo!()),
//         |a, b| (todo!(), todo!()),
//         |a, b| todo!(),
//         |a: O| (todo!(), todo!()),
//         |a, b| (todo!(), todo!()),
//         |a, b| todo!(),
//     );
//     todo!()
// }
//
// pub struct ModelLayerBuilder {
//
// }
//
// pub struct Node<T> {
//     inner: T,
// }
//
// impl<L: LayerBuilder, I> Node<(L, I)> {
//     pub fn new(layer: L, info: I) -> Self {
//         Self {
//             inner: (layer, info),
//         }
//     }
// }
