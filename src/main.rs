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
    // Wrap audio_volume in an Arc<Mutex<>> to share it between the main thread and audio thread
    let audio_volume = Arc::new(Mutex::new(0.0f32));

    // Create audio device and keep in scope so the audio thread continues running
    let _audio_device = prepare_audio(Arc::clone(&audio_volume))?;

    let args: Vec<String> = env::args().collect();

    // Wrap CPU in an Arc<Mutex<>> to share it between the main thread and CPU thread
    let cpu = Arc::new(Mutex::new(Cpu::default()));
    let rom: Rom;

    // Load the specified rom from the first arg or load the fallback rom if no arg
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

    // Spawn a separate thread to run the CPU
    let cpu_clone = Arc::clone(&cpu);
    Cpu::run_concurrent(cpu_clone);

    // Set up the minifb window
    let mut window = Window::new(
        "CHIP-8 Emu",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: Scale::X8,
            scale_mode: ScaleMode::AspectRatioStretch,
            ..WindowOptions::default()
        },
    )?;

    window.set_target_fps(FRAME_RATE);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        {
            // Lock the volume
            let mut volume_lock = audio_volume.lock()?;

            // Lock the CPU and check if sound needs to be played
            let cpu_lock = cpu.lock()?;
            *volume_lock = if cpu_lock.get_sound().clone() > 0 {
                VOLUME_ON
            } else {
                VOLUME_OFF
            };
        }

        {
            // Lock the CPU and update the keyboars
            let mut cpu_lock = cpu.lock()?;
            update_cpu_keyboard(&mut cpu_lock, &window);
        }

        let window_buffer = {
            // Lock the CPU and grab the display buffer
            let cpu_lock = cpu.lock()?;
            display_buffer_to_rgb(cpu_lock.get_display().as_slice())
        };

        // Update the window with the current display buffer.
        window.update_with_buffer(&window_buffer, WIDTH, HEIGHT)?;
    }

    Ok(())
}
