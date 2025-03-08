mod cpu;
pub mod error;
mod helper;
mod rom;

use cpu::{Cpu, HEIGHT, WIDTH};
use error::Error;
use helper::{display_buffer_to_rgb, prepare_audio, update_cpu_keyboard};
use minifb::{Key, Scale, ScaleMode, Window, WindowOptions};
use rom::Rom;
use std::env;
use std::path::Path;
use std::sync::{Arc, Mutex};

const FRAME_RATE: usize = 60;
const VOLUME_ON: f32 = 0.03;
const VOLUME_OFF: f32 = 0.00;

fn main() -> Result<(), Error> {
    let volume_for_audio = Arc::new(Mutex::new(0.0f32));

    // Keep device in scope so the audio thread keeps running
    let _audio_device = prepare_audio(Arc::clone(&volume_for_audio))?;

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
    window.set_target_fps(FRAME_RATE);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        {
            // Lock the volume variable to update it
            let mut vol = volume_for_audio.lock().unwrap();
            // If any key is pressed, set volume to 0.03; otherwise, mute (0.0)

            // Lock the CPU and update its keyboard state
            let mut cpu_lock = cpu.lock()?;
            update_cpu_keyboard(&mut cpu_lock, &window);
            *vol = if cpu_lock.read_sound().clone() > 0 {
                VOLUME_ON
            } else {
                VOLUME_OFF
            };
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
