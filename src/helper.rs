use crate::cpu::Cpu;
use crate::error::Error;
use minifb::{Key, Window};
use std::sync::{Arc, Mutex};
use tinyaudio::prelude::*;

pub fn display_buffer_to_rgb(buffer: &[u8]) -> Vec<u32> {
    let mut pixels = Vec::with_capacity(64 * 32); // WIDTH * HEIGHT
    for &byte in buffer {
        for bit in (0..8).rev() {
            let pixel_on = (byte >> bit) & 1;
            let color = if pixel_on == 1 {
                0xFFFFFFFF
            } else {
                0xFF000000
            };
            pixels.push(color);
        }
    }
    pixels
}

pub fn update_cpu_keyboard(cpu: &mut Cpu, window: &Window) {
    cpu.get_keyboard_mut().clear();
    for key in window.get_keys() {
        match key {
            Key::Key1 => cpu.get_keyboard_mut().key_down(0x1),
            Key::Key2 => cpu.get_keyboard_mut().key_down(0x2),
            Key::Key3 => cpu.get_keyboard_mut().key_down(0x3),
            Key::Key4 => cpu.get_keyboard_mut().key_down(0xC),
            Key::Q => cpu.get_keyboard_mut().key_down(0x4),
            Key::W => cpu.get_keyboard_mut().key_down(0x5),
            Key::E => cpu.get_keyboard_mut().key_down(0x6),
            Key::R => cpu.get_keyboard_mut().key_down(0xD),
            Key::A => cpu.get_keyboard_mut().key_down(0x7),
            Key::S => cpu.get_keyboard_mut().key_down(0x8),
            Key::D => cpu.get_keyboard_mut().key_down(0x9),
            Key::F => cpu.get_keyboard_mut().key_down(0xE),
            Key::Z => cpu.get_keyboard_mut().key_down(0xA),
            Key::X => cpu.get_keyboard_mut().key_down(0x0),
            Key::C => cpu.get_keyboard_mut().key_down(0xB),
            Key::V => cpu.get_keyboard_mut().key_down(0xF),
            _ => {}
        }
    }
}

// TODO Audio scuffed, idk what I'm doing here
pub fn prepare_audio(volume: Arc<Mutex<f32>>) -> Result<OutputDevice, Error> {
    let params = OutputDeviceParameters {
        channels_count: 2,
        sample_rate: 44100,
        channel_sample_count: 4410,
    };

    let device = run_output_device(params, {
        let vol_clone = Arc::clone(&volume);
        let mut clock = 0f32;
        move |data| {
            let vol = *vol_clone.lock().unwrap();
            for samples in data.chunks_mut(params.channels_count) {
                clock = (clock + 1.0) % params.sample_rate as f32;
                let value =
                    (clock * 440.0 * 2.0 * std::f32::consts::PI / params.sample_rate as f32).sin();
                for sample in samples {
                    *sample = value * vol;
                }
            }
        }
    })
    .map_err(|e| Error::Audio(format!("Audio error: {}", e)))?;

    Ok(device)
}
