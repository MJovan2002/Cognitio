use std::{
    fs::{
        read,
        File,
    },
    io::{
        Bytes,
        Read,
        Cursor,
    },
    path::Path,
};
#[cfg(feature = "datasets")]
use std::io::Write;
use tensor::{Tensor, VecProvider};

use crate::datasets::Dataset;

pub struct MNIST<P0: AsRef<Path>, P1: AsRef<Path>, I> {
    path0: P0,
    path1: P1,
    get_training: fn(&P0) -> I,
    get_testing: fn(&P1) -> I,
}

impl<P0: AsRef<Path>, P1: AsRef<Path>> MNIST<P0, P1, File> {
    pub fn online(path0: P0, path1: P1) -> Option<Self> {
        check_and_download(&path0, &path1)?;
        Some(Self {
            path0,
            path1,
            get_training: |p| File::open(p).unwrap(),
            get_testing: |p| File::open(p).unwrap(),
        })
    }
}

impl<P0: AsRef<Path>, P1: AsRef<Path>> MNIST<P0, P1, Cursor<Vec<u8>>> {
    pub fn offline(path0: P0, path1: P1) -> Option<Self> {
        check_and_download(&path0, &path1)?;
        Some(Self {
            path0,
            path1,
            get_training: |p| Cursor::new(read(p).unwrap()),
            get_testing: |p| Cursor::new(read(p).unwrap()),
        })
    }
}

impl<P0: AsRef<Path>, P1: AsRef<Path>, I: Read> Dataset for MNIST<P0, P1, I> {
    type Input = Tensor<u8, VecProvider, 2>;
    type Label = Tensor<u8, VecProvider, 0>;
    type Iter = MNISTIter<I>;

    fn get_training_iter(&self) -> Self::Iter {
        MNISTIter((self.get_training)(&self.path0))
    }

    fn get_testing_iter(&self) -> Self::Iter {
        MNISTIter((self.get_testing)(&self.path1))
    }
}

pub struct FileIter(Bytes<File>);

impl Iterator for FileIter {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(Result::unwrap)
    }
}

pub struct MNISTIter<I>(I);

impl<I: Read> Iterator for MNISTIter<I> {
    type Item = (Tensor<u8, VecProvider, 2>, Tensor<u8, VecProvider, 0>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut image = [0; 784];
        self.0.read_exact(&mut image).ok()?;
        let mut label = [0; 1];
        self.0.read_exact(&mut label).ok()?;
        todo!()
        // Some((Tensor::from(image), Tensor::from(label[0])))
    }
}

fn check_and_download<P0: AsRef<Path>, P1: AsRef<Path>>(path0: P0, path1: P1) -> Option<()> {
    //noinspection HttpUrlsUsage
    #[cfg(feature = "datasets")]
    fn download<P0: AsRef<Path>, P1: AsRef<Path>>(path0: P0, path1: P1) {
        for (path, url0, url1) in [
            (path0.as_ref(), "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"),
            (path1.as_ref(), "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"),
        ] {
            let resp = ureq::get(url0).call().unwrap().into_reader();
            let mut t0 = flate2::read::GzDecoder::new(resp);
            t0.read_exact(&mut [0; 16]).unwrap();

            let resp = ureq::get(url1).call().unwrap().into_reader();
            let mut t1 = flate2::read::GzDecoder::new(resp);
            t1.read_exact(&mut [0; 8]).unwrap();

            let mut output = File::create(path).unwrap();
            let mut buf = [0; 785];
            while let (Ok(_), Ok(_)) = (t0.read_exact(&mut buf[..784]), t1.read_exact(&mut buf[784..])) {
                output.write_all(&buf).unwrap();
            }
        }
    }

    fn check_files<P0: AsRef<Path>, P1: AsRef<Path>>(path0: P0, path1: P1) -> bool {
        let t0 = crc32fast::hash(&read(path0).unwrap());
        let t1 = crc32fast::hash(&read(path1).unwrap());

        (t0 == 1235615393) && (t1 == 2212701242)
    }

    if !check_files(&path0, &path1) {
        #[cfg(feature = "datasets")]
        download(&path0, &path1);
        #[cfg(not(feature = "datasets"))]
        return None;
    }
    Some(())
}
