use std::collections::HashMap;
use rand::distributions::{WeightedIndex, Distribution};

pub struct MarkovProbs {
    name: String,
    probs: HashMap<i32,Vec<MarkovDist>>
}

impl MarkovDist {
    pub fn print(self) {
        for (key, val) in self.probs {
            println!("key: {key} val: {:?}", val);
        }
    }
}

pub struct MarkovDist {
    probs: HashMap<String,Vec<f32>>
}

impl MarkovDist {
    pub fn add(&mut self, value: &str) {
        let key = value.to_string();

        if let Some(x) = self.probs.get_mut(&key) {
            x.push(rand::random::<f32>());
        } else {
            let temp = vec![rand::random::<f32>()];
            self.probs.insert(
                key,
                temp
            );
        }
    }

    pub fn print(self) {
        for (key, val) in self.probs {
            println!("key: {key} val: {:?}", val);
        }
    }
}

fn main() {
    let mut test = MarkovProbs {
        name: "test".to_string(),
        probs: HashMap::new()
    };

    for n in 1..=5 {
        let mut test = MarkovDist {
            probs: HashMap::new()
        };
        test.add("bruh");
        test.add("bruh");
        test.add("bruh");
        test.add("oof");
        test.print();
    }
    
}