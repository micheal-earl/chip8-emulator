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

// TODO proper error types instead of &'static str
fn main() -> Result<(), &'static str> {
    let args: Vec<String> = env::args().collect();

    // Wrap CPU in an Arc<Mutex<>> to share it between threads.
    let cpu = Arc::new(Mutex::new(Cpu::default()));

    // {
    //     // Set up the CPU state before starting the CPU thread.
    //     let mut cpu_lock = cpu.lock().unwrap();
    //     cpu_lock.write_register(0x0000, 5)?;
    //     cpu_lock.write_register(0x0001, 10)?;

    //     let instructions = [
    //         (0x0200, 0x2300), // CALL subroutine at 0x0300
    //         (0x0202, 0x2300), // CALL subroutine at 0x0300
    //         (0x0204, 0x1FFF), // JUMP to 0x1FFF (end of memory)
    //         // Function
    //         (0x0300, 0x8014), // ADD reg[1] to reg[0]
    //         (0x0302, 0x8014), // ADD reg[1] to reg[0]
    //         (0x0304, 0x00EE), // RETURN
    //     ];

    //     cpu_lock.write_instructions_batch(&instructions)?;

    //     for i in 0..1000 {
    //         cpu_lock.write_display(i, true);
    //     }
    // }

    if args.len() > 1 {
        // Load ROM from file provided as command-line argument.
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
        let mut cpu_lock = cpu.lock().unwrap();

        // Load a program that draws three shapes with height 3:
        // 1. U shape at (10,10):
        //    - Load V0 = 10, V1 = 10.
        //    - Set I = 0x300 and draw a sprite with height 3
        //
        // 2. Clipping shape at (60,20):
        //    - Load V0 = 60, V1 = 20
        //    - Set I = 0x310 and draw a sprite with height 3
        //    (Since 60 + 8 > 64, only columns 60–63 will be drawn.)
        //
        // 3. Wrapping shape with starting coordinate off‑screen:
        //    - Load V0 = 70, V1 = 5.
        //    - Since 70 ≥ WIDTH, it wraps (70 % 64 = 6)
        //    - Set I = 0x320 and draw a sprite with height 3
        let instructions = [
            (0x0200, 0x00E0), // CLS
            // U shape at (10,10) with height 3
            (0x0202, 0x600A), // LD V0, 10
            (0x0204, 0x610A), // LD V1, 10
            (0x0206, 0xA300), // LDI 0x300
            (0x0208, 0xD013), // DRW V0,V1,3
            // Clipping shape at (61,21) with height 3
            (0x020A, 0x603D), // LD V0, 61 (0x3D)
            (0x020C, 0x6115), // LD V1, 21 (0x15)
            (0x020E, 0xA310), // LDI 0x310
            (0x0210, 0xD013), // DRW V0,V1,3
            // Wrapping shape with V0=70, V1=5 (70 wraps to 6) with height 3
            (0x0212, 0x6046), // LD V0, 70 (0x46)
            (0x0214, 0x6105), // LD V1, 5  (0x05)
            (0x0216, 0xA320), // LDI 0x320
            (0x0218, 0xD013), // DRW V0,V1,3
            // Halt the CPU
            (0x021A, 0x1FFF), // JP 0x1FFF
        ];
        cpu_lock.write_opcode_batch(&instructions)?;

        // Write sprite data for each shape:

        // At 0x300: Normal shape sprite (3 rows)
        // Row 0: 0x81 -> 10000001
        // Row 1: 0x81 -> 10000001
        // Row 2: 0xFF -> 11111111
        cpu_lock.write_memory(0x300, 0x81)?;
        cpu_lock.write_memory(0x301, 0x81)?;
        cpu_lock.write_memory(0x302, 0xFF)?;

        // At 0x310: Clipping shape sprite (3 rows)
        // Row 0: 0xFF -> 11111111
        // Row 1: 0xFF -> 11111111
        // Row 2: 0xFF -> 11111111
        cpu_lock.write_memory(0x310, 0xFF)?;
        cpu_lock.write_memory(0x311, 0xFF)?;
        cpu_lock.write_memory(0x312, 0xFF)?;

        // At 0x320: Wrapping shape sprite (3 rows)
        // Row 0: 0x3F -> 00111111
        // Row 1: 0x3F -> 00111111
        // Row 2: 0x3F -> 00111111
        cpu_lock.write_memory(0x320, 0x3F)?;
        cpu_lock.write_memory(0x321, 0x3F)?;
        cpu_lock.write_memory(0x322, 0x3F)?;
    }

    // Spawn a separate thread to run the CPU.
    let cpu_clone = Arc::clone(&cpu);
    thread::spawn(move || -> Result<(), CpuError> {
        use std::thread::sleep;
        use std::time::Instant;

        // ~700 Hz => ~1428 microseconds per instruction:
        let cycle_duration = DURATION_700HZ_IN_MICROS;
        let mut next_time = Instant::now() + cycle_duration;

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
            if now < next_time {
                sleep(next_time - now);
            }
            next_time += cycle_duration;
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
