pub mod cpu;
pub mod rom;

use cpu::{Cpu, CpuError, DURATION_60HZ_IN_MICROS, DURATION_700HZ_IN_MICROS, HEIGHT, WIDTH};
use minifb::{Key, Scale, ScaleMode, Window, WindowOptions};
use std::env;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

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
    cpu.reset_keyboard();

    // Keyboard layout

    // Original CHIP-8 Keypad:
    // 1 2 3 C
    // 4 5 6 D
    // 7 8 9 E
    // A 0 B F

    // Emulated CHIP-8 Keypad:
    // 1 2 3 4
    // Q W E R
    // A S D F
    // Z X C V

    // Get the list of pressed keys from the window
    for key in window.get_keys() {
        match key {
            Key::Key1 => cpu.key_down(0x1),
            Key::Key2 => cpu.key_down(0x2),
            Key::Key3 => cpu.key_down(0x3),
            Key::Key4 => cpu.key_down(0xC),
            Key::Q => cpu.key_down(0x4),
            Key::W => cpu.key_down(0x5),
            Key::E => cpu.key_down(0x6),
            Key::R => cpu.key_down(0xD),
            Key::A => cpu.key_down(0x7),
            Key::S => cpu.key_down(0x8),
            Key::D => cpu.key_down(0x9),
            Key::F => cpu.key_down(0xE),
            Key::Z => cpu.key_down(0xA),
            Key::X => cpu.key_down(0x0),
            Key::C => cpu.key_down(0xB),
            Key::V => cpu.key_down(0xF),
            _ => {}
        }
    }
}

fn main() -> Result<(), &'static str> {
    let args: Vec<String> = env::args().collect();

    // Wrap CPU in an Arc<Mutex<>> to share it between threads
    let cpu = Arc::new(Mutex::new(Cpu::default()));

    if args.len() > 1 {
        // Load ROM from file provided as command-line argument
        let rom_path = &args[1];
        println!("Loading ROM: {}", rom_path);
        // TODO Fix awful rom API
        let mut rom = rom::Rom::default();
        rom.open_file(Path::new(rom_path));
        rom.get_instructions()?;

        // Write the ROM bytes into CPU memory starting at 0x200.
        let mut cpu_lock = cpu.lock().unwrap();
        let start_address = 0x200;
        for (i, &byte) in rom.instructions.iter().enumerate() {
            cpu_lock.write_memory((start_address + i) as u16, byte)?;
        }
    } else {
        // TODO Change behavior on no arg or mangled arg
        let mut cpu_lock = cpu.lock().unwrap();

        // Load a program that draws a shape with height 3:
        // 1. U shape at (10,10):
        //    - Load V0 = 10, V1 = 10.
        //    - Set I = 0x300 and draw a sprite with height 3
        let opcodes = [
            (0x0200, 0x00E0), // CLS
            // U shape at (10,10) with height 3
            (0x0202, 0x600A), // LD V0, 10
            (0x0204, 0x610A), // LD V1, 10
            (0x0206, 0xA300), // LDI 0x300
            (0x0208, 0xD013), // DRW V0,V1,3
            (0x021A, 0x1FFF), // JP 0x1FFF
        ];
        cpu_lock.write_opcode_batch(&opcodes)?;

        // Write sprite data for the shape:
        let shape = [(0x0300, 0x81), (0x0301, 0x81), (0x0302, 0xFF)];
        cpu_lock.write_memory_batch(&shape)?;
    }

    // Spawn a separate thread to run the CPU.
    // TODO There is repeat code here that appears in cpu.run()
    // Find a way to reduce repeated code
    let cpu_clone = Arc::clone(&cpu);
    thread::spawn(move || -> Result<(), CpuError> {
        use std::thread::sleep;
        use std::time;

        let cycle_interval = DURATION_700HZ_IN_MICROS;
        let sound_and_delay_interval = DURATION_60HZ_IN_MICROS;
        let mut last_sd_update = time::Instant::now();
        let mut next_cycle = time::Instant::now() + cycle_interval;

        loop {
            {
                // Lock the CPU for just one instruction
                let mut cpu = cpu_clone.lock().unwrap();
                // If step() returns false, exit the loop
                if !cpu.step()? {
                    break;
                }

                // Update timers if 1/60 sec has elapsed
                let now = time::Instant::now();
                if now.duration_since(last_sd_update) >= sound_and_delay_interval {
                    let delay_counter = cpu.read_delay();
                    if delay_counter > 0 {
                        cpu.write_delay(delay_counter - 1);
                    }
                    let sound_counter = cpu.read_sound();
                    if sound_counter > 0 {
                        cpu.write_sound(sound_counter - 1);
                        // TODO Play sound
                    }
                    last_sd_update = now;
                }
            }

            // Sleep until the next cycle
            let now = time::Instant::now();
            if now < next_cycle {
                sleep(next_cycle - now);
            }

            next_cycle += cycle_interval;
        }

        Ok(())
    });

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
            let mut cpu_lock = cpu.lock().unwrap();
            update_cpu_keyboard(&mut cpu_lock, &window);
        }
        // Fetch the display buffer from the CPU.
        let display_buffer = {
            let cpu_lock = cpu.lock().unwrap();
            // TODO maybe slow to clone? idk
            cpu_lock.read_display().clone()
        };
        let window_buffer = display_buffer_to_rgb(&display_buffer);

        // Update the window with the current display buffer.
        window
            .update_with_buffer(&window_buffer, WIDTH, HEIGHT)
            .unwrap();
    }

    Ok(())
}
