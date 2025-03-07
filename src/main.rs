mod cpu;
pub mod error;
mod rom;

use cpu::{Cpu, HEIGHT, WIDTH};
use error::Error;
use minifb::{Key, Scale, ScaleMode, Window, WindowOptions};
use rom::Rom;
use std::env;
use std::path::Path;
use std::sync::{Arc, Mutex};

fn display_buffer_to_rgb(buffer: &[u8]) -> Vec<u32> {
    let mut pixels = Vec::with_capacity(WIDTH * HEIGHT);
    // Each byte in the buffer contains 8 pixels (bits)
    for &byte in buffer {
        // Process bits from most-significant to least-significant
        for bit in (0..8).rev() {
            let pixel_on = (byte >> bit) & 1;
            let color = if pixel_on == 1 {
                0xFFFFFFFF // White with full opacity
            } else {
                0xFF000000 // Black with full opacity
            };
            pixels.push(color);
        }
    }
    pixels
}

fn update_cpu_keyboard(cpu: &mut Cpu, window: &Window) {
    // Clear all keys first
    cpu.keyboard.clear();

    // Get the list of pressed keys from the window
    for key in window.get_keys() {
        match key {
            Key::Key1 => cpu.keyboard.key_down(0x1),
            Key::Key2 => cpu.keyboard.key_down(0x2),
            Key::Key3 => cpu.keyboard.key_down(0x3),
            Key::Key4 => cpu.keyboard.key_down(0xC),
            Key::Q => cpu.keyboard.key_down(0x4),
            Key::W => cpu.keyboard.key_down(0x5),
            Key::E => cpu.keyboard.key_down(0x6),
            Key::R => cpu.keyboard.key_down(0xD),
            Key::A => cpu.keyboard.key_down(0x7),
            Key::S => cpu.keyboard.key_down(0x8),
            Key::D => cpu.keyboard.key_down(0x9),
            Key::F => cpu.keyboard.key_down(0xE),
            Key::Z => cpu.keyboard.key_down(0xA),
            Key::X => cpu.keyboard.key_down(0x0),
            Key::C => cpu.keyboard.key_down(0xB),
            Key::V => cpu.keyboard.key_down(0xF),
            _ => {}
        }
    }
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();

    // Wrap CPU in an Arc<Mutex<>> to share it between threads
    let cpu = Arc::new(Mutex::new(Cpu::default()));
    let rom: Rom;

    // Load the specified rom from the first argu or load the fallback rom if no arg
    if args.len() > 1 {
        rom = rom::Rom::from_path(Path::new(&args[1]))?;
    } else {
        rom = rom::Rom::from_path(Path::new("./roms/test/IBM Logo.ch8"))?;
    }

    // Lock the cpu and load the rom into memory
    {
        let mut cpu_lock = cpu.lock()?;
        cpu_lock.load_rom(rom)?;
    }

    // Spawn a separate thread to run the CPU.
    let cpu_clone = Arc::clone(&cpu);
    Cpu::run_concurrent(cpu_clone);

    // Set up the minifb window.
    let mut window = Window::new(
        "CHIP-8",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X8,
            scale_mode: ScaleMode::AspectRatioStretch,
            ..WindowOptions::default()
        },
    )
    .expect("Unable to open the window");

    // Limit to max ~60 fps update rate.
    window.set_target_fps(60);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        {
            // Lock the CPU and update its keyboard state
            let mut cpu_lock = cpu.lock()?;
            update_cpu_keyboard(&mut cpu_lock, &window);
        }
        // Fetch the display buffer from the CPU.
        let display_buffer = {
            let cpu_lock = cpu.lock()?;
            // TODO maybe slow to clone? idk
            cpu_lock.display.as_slice().clone()
        };
        let window_buffer = display_buffer_to_rgb(&display_buffer);

        // Update the window with the current display buffer.
        window.update_with_buffer(&window_buffer, WIDTH, HEIGHT)?;
    }

    Ok(())
}
