pub mod cpu;
pub mod rom;

use cpu::{Cpu, CpuError, DURATION_700HZ_IN_MICROS, HEIGHT, WIDTH};
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

fn main() -> Result<(), &'static str> {
    let args: Vec<String> = env::args().collect();

    // Wrap CPU in an Arc<Mutex<>> to share it between threads
    let cpu = Arc::new(Mutex::new(Cpu::default()));

    if args.len() > 1 {
        // Load ROM from file provided as command-line argument
        let rom_path = &args[1];
        println!("Loading ROM: {}", rom_path);
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
    let cpu_clone = Arc::clone(&cpu);
    thread::spawn(move || -> Result<(), CpuError> {
        use std::thread::sleep;
        use std::time::Instant;

        let interval = DURATION_700HZ_IN_MICROS;
        let mut time_after_interval = Instant::now() + interval;

        loop {
            {
                // Lock the CPU for just one instruction
                let mut cpu = cpu_clone.lock().unwrap();
                // If step() returns false, exit the loop
                if !cpu.step()? {
                    break;
                }
            }

            // Sleep until the next cycle
            let now = Instant::now();
            if now < time_after_interval {
                sleep(time_after_interval - now);
            }

            time_after_interval += interval;
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
