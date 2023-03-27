#![feature(auto_traits)]
#![feature(allocator_api)]
#![feature(negative_impls)]
#![feature(const_trait_impl)]
#![feature(adt_const_params)]
#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]
#![feature(type_alias_impl_trait)]
#![feature(associated_type_bounds)]
#![feature(array_methods)]
#![feature(associated_type_defaults)]
#![feature(array_zip)]
// #![feature(split_array)]
// #![feature(const_convert)]
// #![feature(iter_advance_by)]
// #![feature(associated_const_equality)]

#![allow(incomplete_features)]
#![forbid(unsafe_code)]
// #![deny(missing_docs)]

//! # Cognito
//! # Blazingly fast and modular ML framework
//!
//! Basic building block of the framework are [`Tensor`]-s and [`Layer`]-s.
//! [`Tensor`]-s are used to pass data around.
//! [`Layer`] is a trait implemented by structs that modify data.
//! [`Layer`] takes one [`Package`] and returns another.
//!
//! [`Layer`]-s are organised into a [`Model`].
//!
//! There are two types of [`Model`]-s:
//! * `Sequential`
//! ```
//! # #![feature(generic_arg_infer)]
//! #
//! # use measure::Measurable;
//! # use tensor::VecProvider;
//! # use cognitio::prelude::*;
//! #
//! # fn main() {
//!     let conv = |x, y, filters| convolution::Builder::new::<_, _, _, { Dim::Static }>()
//!         .kernel_shape([x, y])
//!         // .dilation([2, 2])
//!         .filters(filters)
//!         .activation(Sigmoid::new());
//!
//!     let pool = |x, y| pooling::Builder::new::<_, _, _, { Dim::Static }, { PoolingType::Average }>()
//!         .pool_shape([x, y])
//!         .strides([2, 2]);
//!
//!     let mut m = Model::sequential()
//!         .add_layer(convert::Builder::new::<_, _, VecProvider, _>().activation(Linear::new(1. / 255., 0.)))
//!         .add_layer(reshape::Builder::new().output_shape([28, 28, 1].into()))
//!         .add_layer(conv(5, 5, 10))
//!         .add_layer(pool(2, 2))
//!         .add_layer(conv(3, 3, 10))
//!         .add_layer(pool(2, 2))
//!         .add_layer(dense::Builder::new().activation(Sigmoid::new()).output_shape([16].into()))
//!         .add_layer(dense::Builder::new().activation(Sigmoid::new()).output_shape([10].into()))
//!         // .add_layer(softmax::Builder::new())
//!         .build([28, 28].into());
//! #     let dataset = MNIST::offline("../training", "../testing").unwrap();
//! #     let optimizer = Adam::new(0.9, 0.99, 0.1).batch(10);
//! #     let mut trainer = m.compile(optimizer);
//! #     let (_, t) = (|| trainer.train(1, &dataset, Square::new(), |l| {
//! #         let mut t = [0.; 10];
//! #         t[*l as usize] = 1.;
//! #         todo!()
//! #         // Tensor::from(t)
//! #     })).measure();
//! #     println!("{t:?}");
//! # }
//! ```
//! * `Graph` TODO
//!
//! Once you created a [`Model`] you can construct a [`Dataset`].
//! ```
//! # #![feature(generic_arg_infer)]
//! #
//! # use measure::Measurable;
//! # use tensor::VecProvider;
//! # use cognitio::prelude::*;
//! #
//! # fn main() {
//! #     let conv = |x, y, filters| convolution::Builder::new::<_, _, _, { Dim::Static }>()
//! #         .kernel_shape([x, y])
//! #         // .dilation([2, 2])
//! #         .filters(filters)
//! #         .activation(Sigmoid::new());
//! #
//! #     let pool = |x, y| pooling::Builder::new::<_, _, _, { Dim::Static }, { PoolingType::Average }>()
//! #         .pool_shape([x, y])
//! #         .strides([2, 2]);
//! #
//! #     let mut m = Model::sequential()
//! #         .add_layer(convert::Builder::new::<_, _, VecProvider, _>().activation(Linear::new(1. / 255., 0.)))
//! #         .add_layer(reshape::Builder::new().output_shape([28, 28, 1].into()))
//! #         .add_layer(conv(5, 5, 10))
//! #         .add_layer(pool(2, 2))
//! #         .add_layer(conv(3, 3, 10))
//! #         .add_layer(pool(2, 2))
//! #         .add_layer(dense::Builder::new().activation(Sigmoid::new()).output_shape([16].into()))
//! #         .add_layer(dense::Builder::new().activation(Sigmoid::new()).output_shape([10].into()))
//! #         // .add_layer(softmax::Builder::new())
//! #         .build([28, 28].into());
//!     let dataset = MNIST::offline("../training", "../testing").unwrap();
//! #     let optimizer = Adam::new(0.9, 0.99, 0.1).batch(10);
//! #     let mut trainer = m.compile(optimizer);
//! #     let (_, t) = (|| trainer.train(1, &dataset, Square::new(), |l| {
//! #         let mut t = [0.; 10];
//! #         t[*l as usize] = 1.;
//! #         todo!()
//! #         // Tensor::from(t)
//! #     })).measure();
//! #     println!("{t:?}");
//! # }
//! ```
//! After that choose an [`Optimizer`].
//! ```
//! # #![feature(generic_arg_infer)]
//! #
//! # use measure::Measurable;
//! # use tensor::VecProvider;
//! # use cognitio::prelude::*;
//! #
//! # fn main() {
//! #     let conv = |x, y, filters| convolution::Builder::new::<_, _, _, { Dim::Static }>()
//! #         .kernel_shape([x, y])
//! #         // .dilation([2, 2])
//! #         .filters(filters)
//! #         .activation(Sigmoid::new());
//! #
//! #     let pool = |x, y| pooling::Builder::new::<_, _, _, { Dim::Static }, { PoolingType::Average }>()
//! #         .pool_shape([x, y])
//! #         .strides([2, 2]);
//! #
//! #     let mut m = Model::sequential()
//! #         .add_layer(convert::Builder::new::<_, _, VecProvider, _>().activation(Linear::new(1. / 255., 0.)))
//! #         .add_layer(reshape::Builder::new().output_shape([28, 28, 1].into()))
//! #         .add_layer(conv(5, 5, 10))
//! #         .add_layer(pool(2, 2))
//! #         .add_layer(conv(3, 3, 10))
//! #         .add_layer(pool(2, 2))
//! #         .add_layer(dense::Builder::new().activation(Sigmoid::new()).output_shape([16].into()))
//! #         .add_layer(dense::Builder::new().activation(Sigmoid::new()).output_shape([10].into()))
//! #         // .add_layer(softmax::Builder::new())
//! #         .build([28, 28].into());
//! #     let dataset = MNIST::offline("../training", "../testing").unwrap();
//!     let optimizer = Adam::new(0.9, 0.99, 0.1).batch(10);
//! #     let mut trainer = m.compile(optimizer);
//! #     let (_, t) = (|| trainer.train(1, &dataset, Square::new(), |l| {
//! #         let mut t = [0.; 10];
//! #         t[*l as usize] = 1.;
//! #         todo!()
//! #         // Tensor::from(t)
//! #     })).measure();
//! #     println!("{t:?}");
//! # }
//! ```
//! Finally, create a [`Trainer`] using the [`Model`] and the [`Optimizer`] and train the model on the [`Dataset`].
//! ```
//! # #![feature(generic_arg_infer)]
//! #
//! # use measure::Measurable;
//! # use tensor::VecProvider;
//! # use cognitio::prelude::*;
//! #
//! # fn main() {
//! #     let conv = |x, y, filters| convolution::Builder::new::<_, _, _, { Dim::Static }>()
//! #         .kernel_shape([x, y])
//! #         // .dilation([2, 2])
//! #         .filters(filters)
//! #         .activation(Sigmoid::new());
//! #
//! #     let pool = |x, y| pooling::Builder::new::<_, _, _, { Dim::Static }, { PoolingType::Average }>()
//! #         .pool_shape([x, y])
//! #         .strides([2, 2]);
//! #
//! #     let mut m = Model::sequential()
//! #         .add_layer(convert::Builder::new::<_, _, VecProvider, _>().activation(Linear::new(1. / 255., 0.)))
//! #         .add_layer(reshape::Builder::new().output_shape([28, 28, 1].into()))
//! #         .add_layer(conv(5, 5, 10))
//! #         .add_layer(pool(2, 2))
//! #         .add_layer(conv(3, 3, 10))
//! #         .add_layer(pool(2, 2))
//! #         .add_layer(dense::Builder::new().activation(Sigmoid::new()).output_shape([16].into()))
//! #         .add_layer(dense::Builder::new().activation(Sigmoid::new()).output_shape([10].into()))
//! #         // .add_layer(softmax::Builder::new())
//! #         .build([28, 28].into());
//! #     let dataset = MNIST::offline("../training", "../testing").unwrap();
//! #     let optimizer = Adam::new(0.9, 0.99, 0.1).batch(10);
//!     let mut trainer = m.compile(optimizer);
//!     let (_, t) = (|| trainer.train(1, &dataset, Square::new(), |l| {
//!         let mut t = [0.; 10];
//!         t[*l as usize] = 1.;
//!         todo!()
//!         // Tensor::from(t)
//!     })).measure();
//!     println!("{t:?}");
//! # }
//! ```
//! [`Tensor`]: tensor::Tensor
//! [`Layer`]: self::layers::Layer
//! [`Loss`]: self::losses::Loss
//! [`Model`]: self::model::Model
//! [`Trainer`]: self::trainer::Trainer
//! [`Dataset`]: self::datasets::Dataset
//! [`Optimizer`]: self::optimizers::Optimizer
//! [`Package`]: self::data::Package

pub mod activations;
pub mod callbacks;
pub mod constraints;
pub mod data;
pub mod datasets;
pub mod initializers;
pub mod layers;
pub mod losses;
pub mod metrics;
pub mod model;
pub mod optimizers;
pub mod regularizers;
pub mod schedules;
pub mod trainer;

#[doc(hidden)]
pub mod prelude {
    pub use crate::{
        activations::{
            elu::ELU,
            exp::EXP,
            sigmoid::Sigmoid,
            identity::Identity,
            linear::Linear,
            relu::ReLU,
            softplus::SoftPlus,
            softsign::SoftSign,
            swish::Swish,
            tanh::Tanh,
        },
        callbacks::{},
        constraints::{
            none::None as NoneConst,
            positive::Positive,
        },
        datasets::{
            Dataset,
            mnist::MNIST,
        },
        initializers::{
            constant,
            random::*,
        },
        layers::{
            *,
            convolution::*,
            pooling::PoolingType,
        },
        losses::{
            square::Square,
        },
        model::{
            Model,
            model_tuple::ModelTuple, // todo: remove
        },
        metrics::{},
        optimizers::{
            adadelta::*,
            adagrad::*,
            adam::*,
            adamax::*,
            amsgrad::*,
            mini_batch::*,
            sgd::*,
        },
        regularizers::{
            none::None as NoneReg,
            l1::L1,
            l2::L2,
            l1_l2::L1L2,
        },
        schedules::{
            exponential_decay::ExponentialDecay,
            polynomial_decay::PolynomialDecay,
            inverse_time_decay::InverseTimeDecay,
        },
        trainer::Trainer,
    };
    #[cfg(feature = "distributions")]
    pub use crate::{
        initializers::variance_scaling::*,
        optimizers::noise::*,
    };
}

// todo: add prelude
// todo: encapsulate all model outputs
// todo: add losses to model
// todo: enable no_std
