# Chip-8 Emulator

This project is a basic command-line program that loads and runs chip-8 ROM files.

I am looking for work! Check the contact section!

### Check out a Video!

[](https://youtu.be/ZOGYosR6_As)

---

## About The Project

I started this project to become more familiar with Rust.

Previously, I went through the [nand2tetris](https://www.nand2tetris.org) course, where you build a simple computer from the ground up, starting with NAND gates and ultimately running Tetris on your own custom-built operating system.

I thoroughly enjoyed that course but didn't explore low-level programming or architecture further, as my focus shifted primarily to web development. Now, however, I'm returning to this domain, and the Chip-8 specification seemed like a perfect starting point.

Using Cowgod's technical reference along with a handful of other sources, I began the project. Rust made certain aspects straightforward, especially with tools like pattern matching, though some parts felt unintuitive—and still do. I'm fairly confident that continuing to use the language for larger and more complex projects is the best way to overcome that initial discomfort.

- Multithreading and concurrency still pose a bit of a challenge. I hadn't designed concurrent systems since school, but Rust simplifies much of this complexity.
  - I ended up utilizing three threads: one for the display, another for the sound, and the last for the CPU itself.
- minifb allowed me to easily create a window and display a pixel buffer.
  - Coming from web development, displaying graphics in a native window was unfamiliar territory. My comfort zone has always been HTML/CSS/JavaScript within a web browser, and this project required a completely different approach. Rust has more sophisticated UI frameworks available, but I intentionally chose something simpler and more barebones.
- Sound proved particularly challenging. To be honest, it's still somewhat unclear to me. I incorporated the tinyaudio crate and adapted some example code to make it work.

Structuring the CPU was relatively straightforward and closely mirrored the technical specification. My main objective was to write idiomatic Rust code. I'm not entirely sure I've achieved that high standard, but it was a priority throughout the development process.

### Built With

- Rust
- [minifb](https://github.com/emoon/rust_minifb) by emoon
- [tinyaudio](https://github.com/mrDIMAS/tinyaudio/tree/main) by mrDIMAS

---

## Running the Emulator

Clone or download the repository and follow the instructions below.

### Prerequisites

This program was written in rust and requires cargo to build.

### Installation

```shell
cargo install
```

## Controls

The original Chip-8 keypad has been mapped to the following keys on your keyboard:

| Original CHIP-8 | Emulated CHIP-8 |
| --------------- | --------------- |
| 1 2 3 C         | 1 2 3 4         |
| 4 5 6 D         | Q W E R         |
| 7 8 9 E         | A S D F         |
| A 0 B F         | Z X C V         |

### Controls for Airplane.ch8

- `S` drops the payload.

The goal is to hit the ground with the payload without hitting any of the planes below.

### Controls for Cave.ch8

- `V` starts and resets the game.
- `2` is up
- `Q` is left
- `E` is right
- `S` id down

The goal is to reach the end without hitting the walls.

---

## Usage

To play a game, run the following command from the root directory

```shell
cargo run .\roms\games\Airplane.ch8
```

or:

```shell
cargo run .\roms\games\Cave.ch8
```

---

## Tests

To see unit test output run:

```shell
cargo test
```

or try one of the test roms:

```shell
cargo run .\roms\test\test_opcode.ch8
```

---

## Contact

**Micheal Earl**  
[@m-earl.bsky.social](https://bsky.app/profile/m-earl.bsky.social)  
[LinkedIn](https://www.linkedin.com/in/micheal-earl/)
Email: [michealearldev@gmail.com](mailto:michealearldev@gmail.com)

---

## Acknowledgments

- Corax98 for the test_opcode.ch8 rom
- Cowgod for his Chip-8 technical reference
- kripod for his chip8-roms repo where I found Airplane.ch8 and Cave.ch8
