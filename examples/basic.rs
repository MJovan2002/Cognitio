#![feature(generic_arg_infer)]

use measure::Measurable;
use tensor::Shape;
use cognitio::prelude::*;

fn main() {
    let conv = |x, y, filters| convolution::Builder::new::<_, _, _, { Dim::Static }>()
        .kernel_shape([x, y])
        // .dilation([2, 2])
        .filters(filters)
        .activation(Sigmoid::new());

    let pool = |x, y| pooling::Builder::new::<_, _, _, { Dim::Static }, { PoolingType::Average }>()
        .pool_shape([x, y])
        .strides([2, 2]);

    let mut m = Model::sequential()
        .add_layer(convert::Builder::new().activation(Linear::new(1. / 255., 0.)))
        .add_layer(reshape::Builder::new().output_shape([28, 28, 1].into()))
        .add_layer(conv(5, 5, 10))
        .add_layer(pool(2, 2))
        .add_layer(conv(3, 3, 10))
        .add_layer(pool(2, 2))
        .add_layer(dense::Builder::new().activation(Sigmoid::new()).output_shape([16].into()))
        .add_layer(dense::Builder::new()
            .activation(Sigmoid::new())
            .kernel_initializer(VarianceScaling::normal(0.1, Mode::FanIn))
            .output_shape([10].into()))
        // .add_layer(softmax::Builder::new())
        .build([28, 28].into());

    let dataset = MNIST::offline("../training", "../testing").unwrap();
    let optimizer = Adam::new(0.9, 0.99, 0.1).batch(10);
    let mut trainer = m.compile(optimizer);
    let ((), t) = (|| trainer.train(1, &dataset, Square::new(), |l| {
        let mut t = [0.; 10];
        t[*l as usize] = 1.;
        Shape::new([10]).into_tensor(|i| t[i])
        // Tensor::from(t)
    })).measure();
    println!("{t:?}");
}
