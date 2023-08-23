extern crate clap;

use clap::Parser;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn fib(n: u32) -> u32 {
    if n == 0 || n == 1 {
        1
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

struct Result {
    fib: u32,
    duraiton: Duration,
}

fn run(n: u32) -> Result {
    let start = std::time::Instant::now();
    let f = fib(n);
    let end = std::time::Instant::now();
    let duration = end - start;
    Result {
        fib: f,
        duraiton: duration,
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The failure probability of the distillation
    #[arg(short)]
    n: u32,

    #[arg(long)]
    parallelism: u32,
}

fn main() {
    let args = Args::parse();
    let parallelism = args.parallelism;
    let n = args.n;

    println!("Calculating the nth element of the Fibbonaci sequence multiple times in parallel, where n = {}, parallelism = {}", n, parallelism);

    let mut receivers = Vec::new();
    let start = std::time::Instant::now();
    for _ in 0..parallelism {
        let (tx, rx) = mpsc::channel();
        receivers.push(rx);
        thread::spawn(move || {
            let result = run(n);
            tx.send(result).unwrap();
        });
    }

    for rx in receivers {
        let result = rx.recv().unwrap();
        println!(
            "fib({}) = {} (elapsed: {:?})",
            n, result.fib, result.duraiton
        );
    }
    let total_duration = std::time::Instant::now() - start;
    println!("Elapsed(total) = {:?}", total_duration);
}
