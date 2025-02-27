pub mod cpu;
use cpu::Cpu;
use minifb::{Key, Scale, ScaleMode, Window, WindowOptions};
use std::sync::{Arc, Mutex};
use std::thread;

const WIDTH: usize = 64;
const HEIGHT: usize = 32;

fn display_buffer_to_rgb(buffer: &[u8]) -> Vec<u32> {
    let mut pixels = Vec::with_capacity(WIDTH * HEIGHT);
    // Each byte in the buffer contains 8 pixels (bits).
    for &byte in buffer {
        // Process bits from most-significant to least-significant.
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
    // Wrap CPU in an Arc<Mutex<>> to share it between threads.
    let cpu = Arc::new(Mutex::new(Cpu::default()));

    {
        // Set up the CPU state before starting the CPU thread.
        let mut cpu_lock = cpu.lock().unwrap();
        cpu_lock.write_register(0x0000, 5)?;
        cpu_lock.write_register(0x0001, 10)?;

        let instructions = [
            (0x0200, 0x2300), // CALL subroutine at 0x0300
            (0x0202, 0x2300), // CALL subroutine at 0x0300
            (0x0204, 0x1FFF), // JUMP to 0x1FFF (end of memory)
            // Function
            (0x0300, 0x8014), // ADD reg[1] to reg[0]
            (0x0302, 0x8014), // ADD reg[1] to reg[0]
            (0x0304, 0x00EE), // RETURN
        ];

        cpu_lock.write_instructions_batch(&instructions)?;

        for i in 0..1000 {
            cpu_lock.write_display(i, true);
        }
    }

    // Spawn a separate thread to run the CPU.
    let cpu_clone = Arc::clone(&cpu);
    thread::spawn(move || {
        // This thread will run the CPU at 700Hz.
        cpu_clone.lock().unwrap().run();
    });

    // Set up the minifb window.
    let mut window = Window::new(
        "CHIP-8",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X16,
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
            cpu_lock.read_display().clone()
        };
        let window_buffer = display_buffer_to_rgb(&display_buffer);

        // Update the window with the current display buffer.
        window
            .update_with_buffer(&window_buffer, WIDTH, HEIGHT)
            .unwrap();
    }

    println!(
        "5 + (10 * 2) + (10 * 2) = {}",
        cpu.lock().unwrap().read_register(0).unwrap()
    );

    Ok(())
}
