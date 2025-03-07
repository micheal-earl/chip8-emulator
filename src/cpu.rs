use crate::error::Error;
use std::thread;
use std::time;

/// An OpCode is 16 bits. These bits determine what the cpu executes.
type OpCode = u16;

/// An Instruction is a decoded OpCode.
type Instruction = (u8, u8, u8, u8, u8, u16);

/// A Register is a memory location containing 8 bits.
type Register = u8;

/// A Register16 is a memory location containing 16 bits.
/// This register type is commonly used to store addresses.
type Register16 = u16;

/// An address is a pointer value stored in a Register16.
/// Addresses are actually only 12 bits.
type Address = u16;

/// The Display is a memory region storing bits for the CHIP-8 pixel buffer.
type Display = [Register; 256];

/// The CHIP-8 describes the 16 u8 registers using Vx where x is the register index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u16)]
pub enum RegisterLabel {
    V0 = 0x0,
    V1 = 0x1,
    V2 = 0x2,
    V3 = 0x3,
    V4 = 0x4,
    V5 = 0x5,
    V6 = 0x6,
    V7 = 0x7,
    V8 = 0x8,
    V9 = 0x9,
    VA = 0xA,
    VB = 0xB,
    VC = 0xC,
    VD = 0xD,
    VE = 0xE,
    VF = 0xF,
}

impl TryFrom<u8> for RegisterLabel {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(RegisterLabel::V0),
            1 => Ok(RegisterLabel::V1),
            2 => Ok(RegisterLabel::V2),
            3 => Ok(RegisterLabel::V3),
            4 => Ok(RegisterLabel::V4),
            5 => Ok(RegisterLabel::V5),
            6 => Ok(RegisterLabel::V6),
            7 => Ok(RegisterLabel::V7),
            8 => Ok(RegisterLabel::V8),
            9 => Ok(RegisterLabel::V9),
            10 => Ok(RegisterLabel::VA),
            11 => Ok(RegisterLabel::VB),
            12 => Ok(RegisterLabel::VC),
            13 => Ok(RegisterLabel::VD),
            14 => Ok(RegisterLabel::VE),
            15 => Ok(RegisterLabel::VF),
            _ => Err(Error::Cpu("Invalid register label accessed".to_string())),
        }
    }
}

/// CHIP-8 Display pixel width
pub const WIDTH: usize = 64;

/// CHIP-8 Display pixel height
pub const HEIGHT: usize = 32;

/// CHIP-8 fonts consist of 16 characters, each defined by 5 bytes.
const FONT_DATA: [u8; 80] = [
    0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
    0x20, 0x60, 0x20, 0x20, 0x70, // 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
    0x90, 0x90, 0xF0, 0x10, 0x10, // 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
    0xF0, 0x10, 0x20, 0x40, 0x40, // 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
    0xF0, 0x90, 0xF0, 0x90, 0x90, // A
    0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
    0xF0, 0x80, 0x80, 0x80, 0xF0, // C
    0xE0, 0x90, 0x90, 0x90, 0xE0, // D
    0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
    0xF0, 0x80, 0xF0, 0x80, 0x80, // F
];

pub const DURATION_700HZ_IN_MICROS: time::Duration = time::Duration::from_micros(1_000_000 / 700);
pub const DURATION_60HZ_IN_MICROS: time::Duration = time::Duration::from_micros(1_000_000 / 60);
pub const DURATION_1HZ_IN_MICROS: time::Duration = time::Duration::from_micros(1_000_000);

// TODO Solidify this mapping somewhere for the api?
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

pub struct Cpu {
    registers: [Register; 16],
    memory: [Register; 4096],
    stack: [Register16; 16],
    display: Display, //[[u8; 64]; 32],
    keyboard: [bool; 16],
    delay: Register, // TODO delay timer
    sound: Register, // TODO sound timer
    index: Register16,
    stack_pointer: usize,
    program_counter: usize,
    rng: u8,
    lock: bool,
}

impl Default for Cpu {
    fn default() -> Self {
        // Get the current time in nanoseconds and cast to u8 for our rng
        let nanos = time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .expect("Time went backwards")
            .as_nanos();
        let seed = (nanos & 0xFFFF_FFFF) as u8;

        let mut cpu = Self {
            registers: [0; 16],
            memory: [0; 4096],
            stack: [0; 16],
            display: [0; 256],
            keyboard: [false; 16],
            delay: 0,
            sound: 0,
            index: 0,
            stack_pointer: 0,
            program_counter: 0x0200,
            rng: seed,
            lock: false,
        };

        cpu.write_fonts_to_memory();

        cpu
    }
}

impl Cpu {
    /// Fetches the next instruction from memory at the program_counter address
    fn fetch(&mut self) -> OpCode {
        let pc = self.program_counter;
        let op_high_byte = self.memory[pc] as u16;
        let op_low_byte = self.memory[pc + 1] as u16;

        // TODO: Remove debug
        let out = op_high_byte << 8 | op_low_byte;
        if out != 0 {
            println!("{}, {:#04x}", self.program_counter, &out);
            // println!(
            //     "{}, {}, {}",
            //     self.display[0], self.display[5], self.display[10]
            // )
        }
        // DEBUG ^

        if !self.lock {
            self.program_counter += 2;
        }

        op_high_byte << 8 | op_low_byte
    }

    /// Decodes an OpCode into an instruction
    fn decode(opcode: OpCode) -> Instruction {
        // 0x73EE is an example OpCode
        // In this OpCode, 73 = High Byte, EE = Low Byte
        // and 7 = High Nibble, 3 = Low Nibble, same pattern for EE

        // Var, Bits, Location,                  Description,
        // n    4     low byte, low nibble       Number of bytes
        // x    4     high byte, low nibble      CPU register
        // y    4     low byte, high nibble      CPU register
        // c    4     high byte, high nibble     Opcode group
        // d    4     low byte, low nibble       Opcode subgroup
        // kk   8     low byte, both nibbles     Integer
        // nnn  12    high byte, low nibble      Memory address
        //            and low byte, both nibbles

        let c = ((opcode & 0xF000) >> 12) as u8;
        let x = ((opcode & 0x0F00) >> 8) as u8;
        let y = ((opcode & 0x00F0) >> 4) as u8;
        let d = (opcode & 0x000F) as u8; // op minor
        let kk = (opcode & 0x00FF) as u8;
        let nnn = opcode & 0x0FFF; // addr

        (c, x, y, d, kk, nnn)
    }

    fn execute(&mut self, decoded: Instruction) -> Result<(), Error> {
        let (c, x, y, d, kk, nnn) = decoded;
        let vx = RegisterLabel::try_from(x)?;
        let vy = RegisterLabel::try_from(y)?;
        match (c, x, y, d) {
            (0x0, 0x0, 0x0, 0x0) => return Ok(()), // 0000
            (0x0, 0x0, 0xE, 0x0) => self.cls(),    // 00E0 Clear the screen
            (0x0, 0x0, 0xE, 0xE) => self.ret(),    // 00EE Return from subroutine
            (0x1, _, _, _) => self.jmp(nnn),       // 1nnn Jump to location nnn
            (0x2, _, _, _) => self.call(nnn),      // Call subroutine at location nnn
            (0x3, _, _, _) => self.se(vx, kk),     // 3xkk Skip next instruction if Vx == kk
            (0x4, _, _, _) => self.sne(vx, kk),    // 4xkk Skip next instruvtion if Vx != kk
            (0x5, _, _, 0x0) => {
                self.se_xy(vx, vy)
                // 5xy0 Skip next instruction if Vx == Vy
            }
            (0x6, _, _, _) => self.ld(vx, kk), // 6xkk Write kk to Vx
            (0x7, _, _, _) => self.add(vx, kk), // 7xkk Add kk to Vx, write result to Vx
            (0x8, _, _, _) => match d {
                0x0 => self.ld_vx_into_vy(vx, vy),
                0x1 => self.or_xy(vx, vy),
                0x2 => self.and_xy(vx, vy),
                0x3 => self.xor_xy(vx, vy),
                0x4 => self.add_xy(vx, vy),
                0x5 => self.sub_xy(vx, vy),
                0x6 => self.shr(vx),
                0x7 => self.subn_xy(vx, vy),
                0xE => self.shl(vx),
                _ => {
                    return Err(Error::Cpu(
                        "0x8 instruction with unknown d value".to_string(),
                    ))
                }
            },
            (0x9, _, _, 0x0) => self.sne_xy(vx, vy), // 9xy0 Skip next instruction if Vx != Vy
            (0xA, _, _, _) => self.ldi(nnn),
            (0xB, _, _, _) => self.jmp_x(nnn),
            (0xC, _, _, _) => self.rnd(vx, kk),
            (0xD, _, _, _) => self.drw(vx, vy, d)?, // Dxyn Display n-byte sprite starting at memory location I at (Vx, Vy), set VF = collision.
            // TODO Use nested match statements like the 0x8 instructions?
            (0xE, _, 0x9, 0xE) => self.skp(vx),
            (0xE, _, 0xA, 0x1) => self.sknp(vx),
            (0xF, _, 0x0, 0x7) => self.ld_from_delay(vx),
            (0xF, _, 0x0, 0xA) => self.ld_from_key(vx),
            (0xF, _, 0x1, 0x5) => self.ld_to_delay(vx),
            (0xF, _, 0x1, 0x8) => self.ld_to_sound(vx),
            (0xF, _, 0x1, 0xE) => self.add_i(vx),
            (0xF, _, 0x2, 0x9) => self.ld_i(vx),
            (0xF, _, 0x3, 0x3) => self.ld_bcd(vx),
            (0xF, _, 0x5, 0x5) => self.ld_from_registers(vx),
            (0xF, _, 0x6, 0x5) => self.ld_to_registers(vx),
            _ => return Err(Error::Cpu("Unknown instruction".to_string())),
        }

        Ok(())
    }

    /// Runs the cpu, stepping through instructions until end of memory
    pub fn run(&mut self) -> Result<(), Error> {
        let cycle_interval = DURATION_700HZ_IN_MICROS;
        let sound_and_delay_interval = DURATION_60HZ_IN_MICROS;
        let mut last_sd_update = time::Instant::now();
        let mut next_cycle = time::Instant::now() + cycle_interval;

        loop {
            let s = self.step()?;
            if !s {
                break;
            }

            // Update timers if 1/60 sec has elapsed
            let now = time::Instant::now();
            if now.duration_since(last_sd_update) >= sound_and_delay_interval {
                if self.delay > 0 {
                    self.delay -= 1;
                }
                if self.sound > 0 {
                    self.sound -= 1;
                    // Optionally trigger a beep here if sound > 0
                }
                last_sd_update = now;
            }

            // Sleep until the next cycle
            let now = time::Instant::now();
            if now < next_cycle {
                thread::sleep(next_cycle - now);
            }

            next_cycle += cycle_interval;
        }

        Ok(())
    }

    /// Steps the cpu forward one instruction
    pub fn step(&mut self) -> Result<bool, Error> {
        if self.program_counter >= 4095 {
            return Ok(false); // or return false if you want to exit
        }

        let opcode = self.fetch();
        let instruction = Self::decode(opcode);
        self.execute(instruction)?;

        Ok(true)
    }

    /// 00E0 CLEAR - Clears the display
    fn cls(&mut self) {
        self.display = [0; 256];
    }

    /// 00EE RET - Return from the current sub-routine
    fn ret(&mut self) {
        if self.stack_pointer == 0 {
            panic!("Stack Underflow!");
        }

        self.stack_pointer -= 1;
        let call_addr = self.stack[self.stack_pointer];
        self.program_counter = call_addr as usize;
    }

    /// 1nnn JUMP - Set program counter to address
    fn jmp(&mut self, addr: Address) {
        self.program_counter = addr as usize;
    }

    /// 2nnn CALL - Sub-routine at `addr`
    fn call(&mut self, addr: Address) {
        if self.stack_pointer >= self.stack.len() {
            panic!("Stack Overflow!")
        }

        self.stack[self.stack_pointer] = self.program_counter as u16;
        self.stack_pointer += 1;
        self.program_counter = addr as usize;
    }

    /// 3xkk SE - Skip next instruction if Vx == kk
    fn se(&mut self, vx: RegisterLabel, kk: u8) {
        if self.registers[vx as usize] == kk {
            self.program_counter += 2;
        }
    }

    /// 4xkk SNE - Skip if Vx != kk
    fn sne(&mut self, vx: RegisterLabel, kk: u8) {
        if self.registers[vx as usize] != kk {
            self.program_counter += 2;
        }
    }

    /// 6xy0 SE - Skip next instruction if Vx == Vy
    fn se_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        if self.registers[vx as usize] == self.registers[vy as usize] {
            self.program_counter += 2;
        }
    }

    /// 6xkk LD - Sets the value `kk` into register `Vx`
    fn ld(&mut self, vx: RegisterLabel, kk: u8) {
        self.registers[vx as usize] = kk;
    }

    /// 7xkk ADD - Adds the value `kk` to register `Vx` and stores the sum in 'Vx'
    fn add(&mut self, vx: RegisterLabel, kk: u8) {
        // Use overflowing_add to ensure program does not panic
        (self.registers[vx as usize], _) = self.registers[vx as usize].overflowing_add(kk);
    }

    /// 8xy0 LD - Stores value of Vy in Vx
    fn ld_vx_into_vy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        self.registers[vx as usize] = self.registers[vy as usize];
    }

    /// 8xy1 OR - Bitwise OR on Vy value and Vx value, store result in Vx
    fn or_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        self.registers[vx as usize] |= self.registers[vy as usize];
    }

    /// 8xy2 AND - Bitwise OR on Vy value and Vx value, store result in Vx
    fn and_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        self.registers[vx as usize] &= self.registers[vy as usize];
    }

    /// 8xy3 XOR - Bitwise XOR on Vy value and Vx value, store result in Vx
    fn xor_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        self.registers[vx as usize] ^= self.registers[vy as usize];
    }

    /// 8xy4 ADD - Add Vy to Vx and store sum in Vx
    fn add_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        let vx_value = self.registers[vx as usize];
        let vy_value = self.registers[vy as usize];

        let (result, overflow) = vx_value.overflowing_add(vy_value);
        self.registers[vx as usize] = result;

        if overflow {
            self.registers[0xF] = 1;
        } else {
            self.registers[0xF] = 0;
        }
    }

    /// 8xy5 SUB - Subtract Vy from Vx and store result in Vx
    fn sub_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        let vx_value = self.registers[vx as usize];
        let vy_value = self.registers[vy as usize];

        if vx_value > vy_value {
            self.registers[0xF] = 1;
        } else {
            self.registers[0xF] = 0;
        }

        let (result, _) = vx_value.overflowing_sub(vy_value);
        self.registers[vx as usize] = result;
    }

    /// 8xy6 SHR - If the least-significant bit of Vx is 1, then VF is set to 1, otherwise 0. Then Vx is divided by 2
    fn shr(&mut self, vx: RegisterLabel) {
        let vx_value = self.registers[vx as usize];
        self.registers[0xF] = vx_value & 1;
        self.registers[vx as usize] = vx_value >> 1;
    }

    /// 8xy7 SUBN - If Vy > Vx, then VF is set to 1, otherwise 0. Then Vx is subtracted from Vy, and the results stored in Vx
    fn subn_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        let vx_value = self.registers[vx as usize];
        let vy_value = self.registers[vy as usize];

        if vy_value > vx_value {
            self.registers[0xF] = 1;
        } else {
            self.registers[0xF] = 0;
        }

        let (result, _) = vy_value.overflowing_sub(vx_value);
        self.registers[vx as usize] = result;
    }

    /// 8xyE SHL - If the most-significant bit of Vx is 1, then VF is set to 1, otherwise to 0. Then Vx is multiplied by 2
    fn shl(&mut self, vx: RegisterLabel) {
        let vx_value = self.registers[vx as usize];
        self.registers[0xF] = (vx_value >> 7) & 1;
        self.registers[vx as usize] = vx_value << 1;
    }

    /// 9xy0 SE - Skip next instruction if Vx == Vy
    fn sne_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        if self.registers[vx as usize] != self.registers[vy as usize] {
            self.program_counter += 2;
        }
    }

    /// Annn LDI - Sets the value `nnn` into the index register (I register)
    fn ldi(&mut self, nnn: u16) {
        self.index = nnn;
    }

    /// Bnnn JP - The program counter is set to nnn plus the value of V0.
    fn jmp_x(&mut self, nnn: u16) {
        self.program_counter = nnn as usize + (self.registers[0] as usize);
    }

    /// Cxkk RND - The interpreter generates a random number from 0 to 255, which is then ANDed with the value kk.
    /// The results are stored in Vx.
    fn rnd(&mut self, vx: RegisterLabel, kk: u8) {
        // Update the RNG state using a simple 8-bit linear congruential generator
        // We choose a multiplier of 37 and an increment of 1 which yields a full period mod 256
        self.rng = self.rng.wrapping_mul(37).wrapping_add(1);
        // Use the new state as the random value (0..=255)
        let random_value = self.rng;
        // Store the result of random_value AND kk in Vx
        self.registers[vx as usize] = random_value & kk;
    }

    /// Dxyn DRW - Display d-byte sprite starting at memory location I at (Vx, Vy)  
    /// Sprites are XOR’d onto the display  
    /// If the sprite is partially off-screen, it is clipped (pixels outside are not drawn)  
    /// If the sprite is completely off-screen, it wraps around  
    /// VF is set to 1 if any drawn pixel is erased  
    fn drw(&mut self, vx: RegisterLabel, vy: RegisterLabel, d: u8) -> Result<(), Error> {
        // Get the original coordinates from registers.
        let orig_x = self.read_register(vx)? as usize;
        let orig_y = self.read_register(vy)? as usize;
        let height = d as usize;

        // Determine if the entire sprite is off-screen:
        // We consider it "entirely off-screen" if the starting coordinate is not in bounds
        let wrap = (orig_x >= WIDTH) || (orig_y >= HEIGHT);

        // If wrapping, adjust coordinates by modulo; otherwise, keep them for clipping
        let x_coord = if wrap { orig_x % WIDTH } else { orig_x };
        let y_coord = if wrap { orig_y % HEIGHT } else { orig_y };

        // Reset collision flag.
        self.registers[0xF] = 0;
        let sprite_start = self.index as usize;

        // For each row of the sprite.
        for row in 0..height {
            // In clipping mode, skip rows that fall outside
            if !wrap && (y_coord + row >= HEIGHT) {
                continue;
            }
            let sprite_byte = self.memory[sprite_start + row];

            // Each sprite row is 8 pixels wide.
            for col in 0..8 {
                // In clipping mode, skip columns that fall outside
                if !wrap && (x_coord + col >= WIDTH) {
                    continue;
                }

                let sprite_pixel = (sprite_byte >> (7 - col)) & 1;
                if sprite_pixel == 1 {
                    // For wrapping mode, calculate coordinates modulo WIDTH/HEIGHT
                    let draw_x = if wrap {
                        (x_coord + col) % WIDTH
                    } else {
                        x_coord + col
                    };
                    let draw_y = if wrap {
                        (y_coord + row) % HEIGHT
                    } else {
                        y_coord + row
                    };
                    let pixel_index = (draw_y * WIDTH + draw_x) as u16;

                    let current_pixel = self.get_display_pixel(pixel_index);
                    let new_pixel = current_pixel ^ 1;
                    if current_pixel == 1 && new_pixel == 0 {
                        self.registers[0xF] = 1;
                    }
                    self.write_display(pixel_index, new_pixel == 1)?;
                }
            }
        }

        Ok(())
    }

    /// Ex9E SKP - Skip next instruction if key with the value of Vx is pressed
    /// Checks the keyboard, and if the key corresponding to the value of Vx is currently in the down position, PC is increased by 2
    fn skp(&mut self, vx: RegisterLabel) {
        if self.keyboard[self.registers[vx as usize] as usize] {
            self.program_counter += 2;
        }
    }

    /// ExA1 SKNP - Skip next instruction if key with the value of Vx is not pressed
    /// Checks the keyboard, and if the key corresponding to the value of Vx is currently in the up position, PC is increased by 2
    fn sknp(&mut self, vx: RegisterLabel) {
        if !self.keyboard[self.registers[vx as usize] as usize] {
            self.program_counter += 2;
        }
    }

    /// Fx07 LD - Set Vx = delay timer value
    /// The value of DT is placed into Vx
    fn ld_from_delay(&mut self, vx: RegisterLabel) {
        self.registers[vx as usize] = self.delay;
    }

    /// Fx0A LD - Wait for a key press, store the value of the key in Vx
    /// All execution stops until a key is pressed, then the value of that key is stored in Vx
    fn ld_from_key(&mut self, vx: RegisterLabel) {
        self.lock = true;
        for i in 0..self.keyboard.len() {
            if self.keyboard[i] {
                self.registers[vx as usize] = i as u8;
                self.lock = false;
                break;
            }
        }
    }

    /// Fx15 - LD DT, Vx
    /// Set delay timer = Vx
    /// DT is set equal to the value of Vx
    fn ld_to_delay(&mut self, vx: RegisterLabel) {
        self.delay = self.registers[vx as usize];
    }

    /// Fx18 - LD ST, Vx
    /// Set sound timer = Vx
    /// ST is set equal to the value of Vx
    fn ld_to_sound(&mut self, vx: RegisterLabel) {
        self.sound = self.registers[vx as usize];
    }

    /// Fx1E - ADD I, Vx
    /// Set I = I + Vx
    /// The values of I and Vx are added, and the results are stored in I
    fn add_i(&mut self, vx: RegisterLabel) {
        self.index += self.registers[vx as usize] as u16;
    }

    /// Fx29 - LD F, Vx  
    /// Set I = location of sprite for font digit Vx  
    /// The value of I is set to the location for the hexadecimal sprite corresponding to the value of Vx
    fn ld_i(&mut self, vx: RegisterLabel) {
        let digit = self.registers[vx as usize];
        self.index = (digit as u16) * 5;
    }

    /// Fx33 - LD B, Vx
    /// Store BCD representation of Vx in memory locations I, I+1, and I+2
    /// The interpreter takes the decimal value of Vx, and places the hundreds digit in memory at location in I,
    /// the tens digit at location I+1, and the ones digit at location I+2
    fn ld_bcd(&mut self, vx: RegisterLabel) {
        let value = self.registers[vx as usize];

        // Calculate the BCD digits
        let hundreds = value / 100;
        let tens = (value % 100) / 10;
        let ones = value % 10;

        let index = self.index as usize;

        // Write the digits to memory at addresses I, I+1, and I+2
        self.memory[index] = hundreds;
        self.memory[index + 1] = tens;
        self.memory[index + 2] = ones;
    }

    /// Fx55 LD - Store registers V0 through Vx in memory starting at location I
    /// The interpreter copies the values of registers V0 through Vx into memory, starting at the address in I
    fn ld_from_registers(&mut self, vx: RegisterLabel) {
        let mut index = self.index;
        let stop = vx as u8;
        for i in 0..=stop {
            self.memory[index as usize] = self.registers[i as usize];
            index += 1;
        }
    }

    /// Fx65 LD - Read registers V0 through Vx from memory starting at location I
    /// The interpreter reads values from memory starting at location I into registers V0 through Vx
    fn ld_to_registers(&mut self, vx: RegisterLabel) {
        let mut index = self.index;
        let stop = vx as u8;
        for i in 0..=stop {
            self.registers[i as usize] = self.memory[index as usize];
            index += 1
        }
    }

    /// Write multiple opcodes to memory at several memory locations
    fn write_fonts_to_memory(&mut self) {
        // The font data occupies 80 bytes starting at memory address 0x000.
        // We don't use write_memory_batch here as that would be slower
        let start = 0x000;
        let end = start + FONT_DATA.len();
        self.memory[start..end].copy_from_slice(&FONT_DATA);
    }

    /// Helper to get the value of an individual pixel from the bit-packed display  
    /// The display is stored as 256 u8’s, each holding 8 pixels  
    /// `pixel` is the overall pixel index (0..2047)  
    pub fn get_display_pixel(&self, pixel_index: u16) -> u8 {
        let byte_index = (pixel_index / 8) as usize;
        let bit_index = 7 - (pixel_index % 8);
        (self.display[byte_index] >> bit_index) & 1
    }

    pub fn write_display(&mut self, pixel_index: u16, value: bool) -> Result<(), Error> {
        // Calculate which byte holds the pixel.
        let byte_index = (pixel_index / 8) as usize;
        // Calculate the bit position within that byte.
        // We assume bit 7 is the leftmost pixel, so we subtract the remainder from 7.
        let bit_index = 7 - (pixel_index % 8);

        // Check if the byte index is valid for our display buffer.
        if byte_index >= self.display.len() {
            return Err(Error::Cpu(
                "Byte index out of bounds of display buffer.".to_string(),
            ));
        }

        if value {
            // Set the bit to 1 to turn the pixel on.
            self.display[byte_index] |= 1 << bit_index;
        } else {
            // Clear the bit to 0 to turn the pixel off.
            self.display[byte_index] &= !(1 << bit_index);
        }

        Ok(())
    }

    /// Read the address in the index (I) register
    pub fn read_index_register(&self) -> &Register16 {
        todo!()
    }

    /// Return a reference to the display memory region
    pub fn read_display(&self) -> &Display {
        &self.display
    }

    /// Returns a copy of the value stored in the provided register
    pub fn read_register(&self, register_label: RegisterLabel) -> Result<Register, Error> {
        if let Some(value) = self.registers.get(register_label as usize).copied() {
            Ok(value)
        } else {
            Err(Error::Cpu("Out of bounds".to_string()))
        }
    }

    /// Write a new value to Vx
    pub fn write_register(
        &mut self,
        register_label: RegisterLabel,
        value: u8,
    ) -> Result<(), Error> {
        let register_label_as_usize = register_label as usize;
        let register_is_in_bounds = register_label_as_usize < self.registers.len();

        if register_is_in_bounds {
            self.registers[register_label_as_usize] = value;
            Ok(())
        } else {
            Err(Error::Cpu("Register index out of bounds".to_string()))
        }
    }

    /// Returns a copy of the value stored in the provided memory address
    pub fn read_memory(&self, address: Address) -> Result<Register, Error> {
        if let Some(value) = self.memory.get(address as usize).copied() {
            Ok(value)
        } else {
            Err(Error::Cpu("Out of bounds".to_string()))
        }
    }

    /// Write a new value to a memory location
    pub fn write_memory(&mut self, address: Address, value: u8) -> Result<(), Error> {
        let index = address as usize;
        if index < self.memory.len() {
            self.memory[index] = value;
            Ok(())
        } else {
            Err(Error::Cpu("Memory address out of bounds".to_string()))
        }
    }

    /// Write multiple new values to several memory locations
    pub fn write_memory_batch(
        &mut self,
        addresses_and_values: &[(Address, u8)],
    ) -> Result<(), Error> {
        for &(address, value) in addresses_and_values {
            self.write_memory(address, value)?;
        }
        Ok(())
    }

    /// Write an opcode to memory. Opcodes are 2 registers long
    pub fn write_opcode(&mut self, address: Address, opcode: OpCode) -> Result<(), Error> {
        let index = address as usize;
        if index % 2 != 0 {
            return Err(Error::Cpu(
                "Cannot insert opcode at odd memory address. Opcodes are 2 registers long."
                    .to_string(),
            ));
        }
        if index + 1 >= self.memory.len() {
            return Err(Error::Cpu("Memory address out of bounds".to_string()));
        }
        // Split the instruction into high and low bytes.
        self.memory[index] = (opcode >> 8) as u8;
        self.memory[index + 1] = (opcode & 0xFF) as u8;
        Ok(())
    }

    pub fn write_opcode_batch(&mut self, opcodes: &[(Address, OpCode)]) -> Result<(), Error> {
        for &(address, opcode) in opcodes {
            self.write_opcode(address, opcode)?;
        }
        Ok(())
    }

    pub fn reset_keyboard(&mut self) {
        self.keyboard = [false; 16];
    }

    pub fn key_down(&mut self, key: u8) {
        self.keyboard[key as usize] = true;
    }

    pub fn read_delay(&mut self) -> u8 {
        self.delay
    }

    pub fn read_sound(&mut self) -> u8 {
        self.sound
    }

    pub fn write_delay(&mut self, value: u8) {
        self.delay = value;
    }

    pub fn write_sound(&mut self, value: u8) {
        self.sound = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_xy_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        // Write initial register values
        cpu.write_register(RegisterLabel::V0, 5)?;
        cpu.write_register(RegisterLabel::V1, 10)?;
        cpu.write_register(RegisterLabel::V2, 15)?;
        cpu.write_register(RegisterLabel::V3, 7)?;

        // Write opcodes into memory using write_memory_batch.
        let opcodes = [
            (0x0200, 0x8014), // 8014: ADD V1 to V0
            (0x0202, 0x8024), // 8024: ADD V2 to V0
            (0x0204, 0x8034), // 8034: ADD V3 to V0
            (0x0206, 0x1FFF), // 1nnn Jump to 0xFFF
        ];

        cpu.write_opcode_batch(&opcodes)?;

        cpu.run()?;

        assert_eq!(cpu.read_register(RegisterLabel::V0).unwrap(), 37);

        Ok(())
    }

    #[test]
    fn add_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        // Set an initial value in register V1
        cpu.write_register(RegisterLabel::V1, 20)?;

        let opcodes = [
            (0x0200, 0x7105), // 7xkk Add 0x05 to register V1 (20 + 5 = 25)
            (0x0202, 0x1FFF), // 1nnn Jump to 0xFFF
        ];

        cpu.write_opcode_batch(&opcodes)?;

        cpu.run()?;

        // Verify that register V1 now holds the value 25.
        assert_eq!(cpu.read_register(RegisterLabel::V1).unwrap(), 25);

        Ok(())
    }

    #[test]
    fn jump_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        cpu.write_register(RegisterLabel::V0, 5)?;
        cpu.write_register(RegisterLabel::V1, 7)?;

        let opcodes = [
            (0x0200, 0x1300), // 1nnn JUMP to nnn
            (0x0300, 0x8014), // 8014 ADD V1 to V0
            (0x0302, 0x1FFF), // 1nnn JUMP to nnn, in this case 0xFFF is the end of memory
        ];

        cpu.write_opcode_batch(&opcodes)?;

        cpu.run()?;

        assert_eq!(cpu.read_register(RegisterLabel::V0).unwrap(), 12);

        Ok(())
    }

    #[test]
    fn call_and_ret_operations() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        // Write initial register values
        cpu.write_register(RegisterLabel::V0, 5)?;
        cpu.write_register(RegisterLabel::V1, 10)?;

        let opcodes = [
            (0x0200, 0x2300), // 2nnn CALL subroutine at addr 0x300
            (0x0202, 0x2300), // 2nnn CALL subroutine at addr 0x300
            (0x0204, 0x1FFF), // 1nnn JUMP to nnn, in this case 0xFFF is the end of memory
            // Function
            (0x0300, 0x8014), // 8014 ADD reg[1] to reg[0]
            (0x0302, 0x8014), // 8014 ADD reg[1] to reg[0]
            (0x0304, 0x00EE), // RETURN
        ];

        cpu.write_opcode_batch(&opcodes)?;

        cpu.run()?;

        assert_eq!(cpu.read_register(RegisterLabel::V0).unwrap(), 45);

        Ok(())
    }

    #[test]
    fn ld_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        let opcodes = [
            (0x0200, 0x63AB), // 6xkk load V3 with 0xAB.
            (0x0202, 0x1FFF), // 1nnn jump to 0xFFF
        ];

        cpu.write_opcode_batch(&opcodes)?;

        cpu.run()?;

        assert_eq!(cpu.read_register(RegisterLabel::V3).unwrap(), 0xAB);

        Ok(())
    }

    #[test]
    fn ldi_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        let opcodes = [
            (0x0200, 0xA123), // Annn load I register with 0x123
            (0x0202, 0x1FFF), // 1nnn jump to 0xFFF
        ];

        cpu.write_opcode_batch(&opcodes)?;

        cpu.run()?;

        assert_eq!(cpu.index, 0x123);

        Ok(())
    }

    #[test]
    fn se_operation() -> Result<(), Error> {
        // Test 3xkk: Skip next instruction if Vx equals kk
        let mut cpu = Cpu::default();
        cpu.write_register(RegisterLabel::V0, 10)?;
        // If V0 == 10, the next instruction should be skipped
        let opcodes = [
            (0x0200, 0x300A), // 3xkk: if V0 == 0x0A, skip next instruction
            (0x0202, 0x6005), // 6xkk: would load 5 into V0 if executed
            (0x0204, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;
        // V0 should remain 10 because the load of 5 was skipped
        assert_eq!(cpu.read_register(RegisterLabel::V0)?, 10);
        Ok(())
    }

    #[test]
    fn sne_operation() -> Result<(), Error> {
        // Test 4xkk: Skip next instruction if Vx does NOT equal kk
        let mut cpu = Cpu::default();
        cpu.write_register(RegisterLabel::V0, 10)?;
        // Since 10 != 5, the next instruction should be skipped
        let opcodes = [
            (0x0200, 0x4005), // 4xkk: if V0 != 0x05, skip next instruction
            (0x0202, 0x6003), // 6xkk: would load 3 into V0 if executed
            (0x0204, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;
        // V0 should remain 10 because the load was skipped
        assert_eq!(cpu.read_register(RegisterLabel::V0)?, 10);
        Ok(())
    }

    #[test]
    fn se_xy_operation() -> Result<(), Error> {
        // Test 5xy0: Skip next instruction if Vx equals Vy
        let mut cpu = Cpu::default();
        cpu.write_register(RegisterLabel::V0, 20)?;
        cpu.write_register(RegisterLabel::V1, 20)?;
        // Since V0 == V1, the next instruction should be skipped
        let opcodes = [
            (0x0200, 0x5010), // 5xy0: if V0 == V1, skip next instruction
            (0x0202, 0x6005), // 6xkk: would load 5 into V0
            (0x0204, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;
        // V0 should still be 20
        assert_eq!(cpu.read_register(RegisterLabel::V0)?, 20);
        Ok(())
    }

    #[test]
    fn sne_xy_operation() -> Result<(), Error> {
        // Test 9xy0: Skip next instruction if Vx does NOT equal Vy
        let mut cpu = Cpu::default();
        cpu.write_register(RegisterLabel::V0, 15)?;
        cpu.write_register(RegisterLabel::V1, 20)?;
        // Since V0 != V1, the next instruction should be skipped
        let opcodes = [
            (0x0200, 0x9010), // 9xy0: if V0 != V1, skip next instruction
            (0x0202, 0x6099), // 6xkk: would load 0x99 into V0
            (0x0204, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;
        // V0 should remain 15
        assert_eq!(cpu.read_register(RegisterLabel::V0)?, 15);
        Ok(())
    }

    #[test]
    fn ld_vx_into_vy_operation() -> Result<(), Error> {
        // Test 8xy0: Set Vx = Vy
        let mut cpu = Cpu::default();
        cpu.write_register(RegisterLabel::V1, 42)?;
        let opcodes = [
            (0x0200, 0x8010), // 8xy0: copy value from V1 to V0
            (0x0202, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;
        assert_eq!(cpu.read_register(RegisterLabel::V0)?, 42);
        Ok(())
    }

    #[test]
    fn or_xy_operation() -> Result<(), Error> {
        // Test 8xy1: Set Vx = Vx OR Vy
        let mut cpu = Cpu::default();
        cpu.write_register(RegisterLabel::V0, 0xA)?; // 0b1010
        cpu.write_register(RegisterLabel::V1, 0x5)?; // 0b0101
        let opcodes = [
            (0x0200, 0x8011), // 8xy1: V0 = V0 OR V1
            (0x0202, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;
        // 0b1010 OR 0b0101 = 0b1111 (0xF)
        assert_eq!(cpu.read_register(RegisterLabel::V0)?, 0xF);
        Ok(())
    }

    #[test]
    fn and_xy_operation() -> Result<(), Error> {
        // Test 8xy2: Set Vx = Vx AND Vy
        let mut cpu = Cpu::default();
        cpu.write_register(RegisterLabel::V0, 0xA)?; // 0b1010
        cpu.write_register(RegisterLabel::V1, 0x6)?; // 0b0110
        let opcodes = [
            (0x0200, 0x8012), // 8xy2: V0 = V0 AND V1
            (0x0202, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;
        // 0b1010 AND 0b0110 = 0b0010 (0x2)
        assert_eq!(cpu.read_register(RegisterLabel::V0)?, 0x2);
        Ok(())
    }

    #[test]
    fn xor_xy_operation() -> Result<(), Error> {
        // Test 8xy3: Set Vx = Vx XOR Vy
        let mut cpu = Cpu::default();
        cpu.write_register(RegisterLabel::V0, 0xA)?; // 0b1010
        cpu.write_register(RegisterLabel::V1, 0x6)?; // 0b0110
        let opcodes = [
            (0x0200, 0x8013), // 8xy3: V0 = V0 XOR V1
            (0x0202, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;
        // 0b1010 XOR 0b0110 = 0b1100 (0xC)
        assert_eq!(cpu.read_register(RegisterLabel::V0)?, 0xC);
        Ok(())
    }

    #[test]
    fn sub_xy_operation() -> Result<(), Error> {
        // Test case where Vx > Vy
        {
            let mut cpu = Cpu::default();
            cpu.write_register(RegisterLabel::V0, 10)?;
            cpu.write_register(RegisterLabel::V1, 5)?;
            let opcodes = [
                (0x0200, 0x8015), // 8xy5: V0 = V0 - V1 using new sub operation
                (0x0202, 0x1FFF), // Jump to halt
            ];
            cpu.write_opcode_batch(&opcodes)?;
            cpu.run()?;
            // V0 should be 5 and VF should be set to 1
            assert_eq!(cpu.read_register(RegisterLabel::V0)?, 5);
            assert_eq!(cpu.read_register(RegisterLabel::VF)?, 1);
        }

        // Test case where Vx <= Vy
        {
            let mut cpu = Cpu::default();
            cpu.write_register(RegisterLabel::V0, 5)?;
            cpu.write_register(RegisterLabel::V1, 10)?;
            let opcodes = [
                (0x0200, 0x8015), // 8xy5: V0 = V0 - V1 using new sub operation
                (0x0202, 0x1FFF), // Jump to halt
            ];
            cpu.write_opcode_batch(&opcodes)?;
            cpu.run()?;
            // V0 should be 251 (5 - 10 with wrapping) and VF should be set to 0
            assert_eq!(cpu.read_register(RegisterLabel::V0)?, 251);
            assert_eq!(cpu.read_register(RegisterLabel::VF)?, 0);
        }
        Ok(())
    }

    #[test]
    fn cls_operation() -> Result<(), Error> {
        // Test 00E0: Clear the display
        let mut cpu = Cpu::default();
        // Fill the display with nonzero data
        for byte in cpu.display.iter_mut() {
            *byte = 0xFF;
        }
        let opcodes = [
            (0x0200, 0x00E0), // CLS instruction
            (0x0202, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;
        // Verify that the entire display is cleared (all zeros)
        for &byte in cpu.read_display().iter() {
            assert_eq!(byte, 0);
        }
        Ok(())
    }

    #[test]
    fn subn_xy_operation_no_borrow() -> Result<(), Error> {
        // Test 8xy7 SUBN: if Vy > Vx then VF is set to 1 and Vx becomes Vy - Vx
        let mut cpu = Cpu::default();
        // Set V0 to 5 and V1 to 10
        cpu.write_register(RegisterLabel::V0, 5)?;
        cpu.write_register(RegisterLabel::V1, 10)?;
        // Expected: VF should be set to 1 and V0 becomes 10 - 5 = 5
        let opcodes = [
            (0x0200, 0x8017), // 8xy7: for x = 0 and y = 1, perform V0 = V1 - V0
            (0x0202, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;
        assert_eq!(cpu.read_register(RegisterLabel::VF)?, 1);
        assert_eq!(cpu.read_register(RegisterLabel::V0)?, 5);
        Ok(())
    }

    #[test]
    fn subn_xy_operation_with_borrow() -> Result<(), Error> {
        // Test 8xy7 SUBN: if Vy <= Vx then VF is set to 0 and Vx becomes Vy - Vx (with wrapping)
        let mut cpu = Cpu::default();
        // Set V0 to 10 and V1 to 5
        cpu.write_register(RegisterLabel::V0, 10)?;
        cpu.write_register(RegisterLabel::V1, 5)?;
        // Expected: VF should be set to 0 and V0 becomes 5 - 10, which in wrapping arithmetic yields 251
        let opcodes = [
            (0x0200, 0x8017), // 8xy7: for x = 0 and y = 1, perform V0 = V1 - V0
            (0x0202, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;
        assert_eq!(cpu.read_register(RegisterLabel::VF)?, 0);
        assert_eq!(cpu.read_register(RegisterLabel::V0)?, 251);
        Ok(())
    }

    #[test]
    fn shl_operation() -> Result<(), Error> {
        // Test case 1: Use V0 with MSB set
        // Program:
        // 6090: LD V0, 0x90    (set V0 = 0x90)
        // 800E: SHL V0        (shift V0 left; VF gets MSB)
        // 1FFF: JP 0xFFF      (halt)
        let opcodes_v0 = [
            (0x0200, 0x6090), // LD V0, 0x90
            (0x0202, 0x800E), // SHL V0
            (0x0204, 0x1FFF), // Halt
        ];
        {
            let mut cpu = Cpu::default();
            cpu.write_opcode_batch(&opcodes_v0)?;
            cpu.run()?;
            // 0x90 (1001 0000) shifted left becomes 0x20 and VF should be 1 since the MSB was 1
            assert_eq!(cpu.read_register(RegisterLabel::V0)?, 0x20);
            assert_eq!(cpu.read_register(RegisterLabel::VF)?, 1);
        }

        // Test case 2: Use V1 with MSB not set
        // Program:
        // 6130: LD V1, 0x30    (set V1 = 0x30)
        // 811E: SHL V1        (shift V1 left; VF gets MSB)
        // 1FFF: JP 0xFFF      (halt)
        let opcodes_v1 = [
            (0x0200, 0x6130), // LD V1, 0x30
            (0x0202, 0x811E), // SHL V1
            (0x0204, 0x1FFF), // Halt
        ];
        {
            let mut cpu = Cpu::default();
            cpu.write_opcode_batch(&opcodes_v1)?;
            cpu.run()?;
            // 0x30 (0011 0000) shifted left becomes 0x60 and VF should be 0 since the MSB was 0
            assert_eq!(cpu.read_register(RegisterLabel::V1)?, 0x60);
            assert_eq!(cpu.read_register(RegisterLabel::VF)?, 0);
        }
        Ok(())
    }

    #[test]
    fn shr_operation() -> Result<(), Error> {
        // Test case 1: Use V0 where LSB is 1
        // Program:
        // 6005: LD V0, 0x05    (set V0 = 5; binary 0101)
        // 8006: SHR V0        (shift V0 right; VF gets LSB)
        // 1FFF: JP 0xFFF      (halt)
        let opcodes_v0 = [
            (0x0200, 0x6005), // LD V0, 0x05
            (0x0202, 0x8006), // SHR V0
            (0x0204, 0x1FFF), // Halt
        ];
        {
            let mut cpu = Cpu::default();
            cpu.write_opcode_batch(&opcodes_v0)?;
            cpu.run()?;
            // 5 >> 1 = 2 and VF becomes 5 & 1 = 1 (since 5 is 0101 in binary)
            assert_eq!(cpu.read_register(RegisterLabel::V0)?, 2);
            assert_eq!(cpu.read_register(RegisterLabel::VF)?, 1);
        }

        // Test case 2: Use V1 where LSB is 0
        // Program:
        // 6104: LD V1, 0x04    (set V1 = 4; binary 0100)
        // 8106: SHR V1        (shift V1 right; VF gets LSB)
        // 1FFF: JP 0xFFF      (halt)
        let opcodes_v1 = [
            (0x0200, 0x6104), // LD V1, 0x04
            (0x0202, 0x8106), // SHR V1
            (0x0204, 0x1FFF), // Halt
        ];
        {
            let mut cpu = Cpu::default();
            cpu.write_opcode_batch(&opcodes_v1)?;
            cpu.run()?;
            // 4 >> 1 = 2 and VF becomes 4 & 1 = 0 (since 4 is 0100 in binary)
            assert_eq!(cpu.read_register(RegisterLabel::V1)?, 2);
            assert_eq!(cpu.read_register(RegisterLabel::VF)?, 0);
        }
        Ok(())
    }

    #[test]
    fn ld_from_registers_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        // Program:
        // 6000: LD V0, 0       (set V0 = 0)
        // 6102: LD V1, 2       (set V1 = 2)
        // 6204: LD V2, 4       (set V2 = 4)
        // 6306: LD V3, 6       (set V3 = 6)
        // 6408: LD V4, 8       (set V4 = 8)
        // 650A: LD V5, 10      (set V5 = 10)
        // A0C8: LDI 0x0C8     (set I = 200)
        // F555: LD [I], V5   (store registers V0 through V5 into memory)
        // 1FFF: JP 0xFFF     (halt)
        let opcodes = [
            (0x0200, 0x6000),
            (0x0202, 0x6102),
            (0x0204, 0x6204),
            (0x0206, 0x6306),
            (0x0208, 0x6408),
            (0x020A, 0x650A),
            (0x020C, 0xA0C8),
            (0x020E, 0xF555),
            (0x0210, 0x1FFF),
        ];
        cpu.write_opcode_batch(&opcodes)?;

        cpu.run()?;

        // Verify that memory from I (200) to I + 5 contains registers V0 through V5
        for i in 0..=5 {
            assert_eq!(cpu.memory[200 + i as usize], cpu.registers[i as usize])
        }

        // Verify that the next memory cell remains unchanged (should be 0)
        assert_eq!(cpu.memory[206], 0);

        Ok(())
    }

    #[test]
    fn ld_to_registers_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        // Preload memory values at address 150 with known data
        // Memory at 150 to 155 should be: 11, 22, 33, 44, 55, 66
        cpu.write_memory(150, 11)?;
        cpu.write_memory(151, 22)?;
        cpu.write_memory(152, 33)?;
        cpu.write_memory(153, 44)?;
        cpu.write_memory(154, 55)?;
        cpu.write_memory(155, 66)?;

        // Program:
        // A096: LDI 0x096    (set I = 150)
        // F565: LD V0-V5, [I] (read registers V0 through V5 from memory)
        // 1FFF: JP 0xFFF     (halt)
        let opcodes = [(0x0200, 0xA096), (0x0202, 0xF565), (0x0204, 0x1FFF)];
        cpu.write_opcode_batch(&opcodes)?;

        cpu.run()?;

        // Verify that registers V0 through V5 have been loaded with memory values from I to I+5
        assert_eq!(cpu.registers[0], 11);
        assert_eq!(cpu.registers[1], 22);
        assert_eq!(cpu.registers[2], 33);
        assert_eq!(cpu.registers[3], 44);
        assert_eq!(cpu.registers[4], 55);
        assert_eq!(cpu.registers[5], 66);

        // Verify that register V6 remains unchanged (should be 0)
        assert_eq!(cpu.registers[6], 0);

        Ok(())
    }

    #[test]
    fn ld_bcd_operation() -> Result<(), Error> {
        // Test Fx33 for register V1 with value 123
        // 0x200: 0x617B -> LD V1, 0x7B (set V1 to 123)
        // 0x202: 0xA12C -> LDI 0x12C (set I to 300 decimal)
        // 0x204: 0xF133 -> Fx33 (store BCD of V1 in memory at I, I+1, I+2)
        // 0x206: 0x1FFF -> JP 0xFFF (halt execution)
        let mut cpu = Cpu::default();
        let opcodes_v1 = [
            (0x0200, 0x617B),
            (0x0202, 0xA12C),
            (0x0204, 0xF133),
            (0x0206, 0x1FFF),
        ];
        cpu.write_opcode_batch(&opcodes_v1)?;
        cpu.run()?;

        // Read memory using the API to verify the BCD digits for 123 are stored correctly
        let hundreds = cpu.read_memory(300)?;
        let tens = cpu.read_memory(301)?;
        let ones = cpu.read_memory(302)?;
        assert_eq!(hundreds, 1);
        assert_eq!(tens, 2);
        assert_eq!(ones, 3);

        // Test Fx33 for register V2 with value 0
        // 0x200: 0x6200 -> LD V2, 0x00 (set V2 to 0)
        // 0x202: 0xA136 -> LDI 0x136 (set I to 310 decimal)
        // 0x204: 0xF233 -> Fx33 (store BCD of V2 in memory at I, I+1, I+2)
        // 0x206: 0x1FFF -> JP 0xFFF (halt execution)
        let mut cpu = Cpu::default();
        let opcodes_v2 = [
            (0x0200, 0x6200),
            (0x0202, 0xA136),
            (0x0204, 0xF233),
            (0x0206, 0x1FFF),
        ];
        cpu.write_opcode_batch(&opcodes_v2)?;
        cpu.run()?;

        let hundreds = cpu.read_memory(310)?;
        let tens = cpu.read_memory(311)?;
        let ones = cpu.read_memory(312)?;
        assert_eq!(hundreds, 0);
        assert_eq!(tens, 0);
        assert_eq!(ones, 0);

        // Test Fx33 for register V3 with value 255
        // 0x200: 0x63FF -> LD V3, 0xFF (set V3 to 255)
        // 0x202: 0xA144 -> LDI 0x144 (set I to 324 decimal)
        // 0x204: 0xF333 -> Fx33 (store BCD of V3 in memory at I, I+1, I+2)
        // 0x206: 0x1FFF -> JP 0xFFF (halt execution)
        let mut cpu = Cpu::default();
        let opcodes_v3 = [
            (0x0200, 0x63FF),
            (0x0202, 0xA144),
            (0x0204, 0xF333),
            (0x0206, 0x1FFF),
        ];
        cpu.write_opcode_batch(&opcodes_v3)?;
        cpu.run()?;

        let hundreds = cpu.read_memory(324)?;
        let tens = cpu.read_memory(325)?;
        let ones = cpu.read_memory(326)?;
        assert_eq!(hundreds, 2);
        assert_eq!(tens, 5);
        assert_eq!(ones, 5);

        Ok(())
    }

    #[test]
    fn test_rnd_opcode() -> Result<(), Error> {
        // create a CPU instance
        let mut cpu = Cpu::default();
        // set a known RNG seed so the output is predictable
        cpu.rng = 123;
        // write opcodes to run the RND instruction on V0 with mask 0xFF
        // 0xC0FF: RND V0, 0xFF (generate a random number and AND with 0xFF, storing the result in V0)
        // 0x1FFF: JP 0xFFF (halt execution)
        let opcodes = [(0x0200, 0xC0FF), (0x0202, 0x1FFF)];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run()?;

        // given the LCG update, new rng = 123 * 37 + 1 = 4552
        // the value stored in V0 is 4552 & 0xFF which equals 200
        let expected: u8 = 200;
        let v0 = cpu.read_register(RegisterLabel::V0)?;
        assert_eq!(v0, expected, "Expected V0 to be {}, got {}", expected, v0);
        Ok(())
    }

    #[test]
    fn jmp_x_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // LD V0, 5
        cpu.write_opcode_batch(&[(0x0200, 0x6005)])?;
        // JMP X: Jump to address 0x0210 using V0 value (0xB20B because 0x20B + 5 = 0x210)
        cpu.write_opcode_batch(&[(0x0202, 0xB20B)])?;
        // At address 0x0210, load V1 with 0xAA and then halt
        cpu.write_opcode_batch(&[(0x0210, 0x61AA), (0x0212, 0x1FFF)])?;
        cpu.run()?;
        assert_eq!(cpu.read_register(RegisterLabel::V1)?, 0xAA);
        Ok(())
    }

    #[test]
    fn skp_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // LD V1, 3 so that V1 holds the key index to check
        cpu.write_opcode_batch(&[(0x0200, 0x6103)])?;
        // Set keyboard[3] to true to simulate key press
        cpu.keyboard[3] = true;
        // SKP V1: if key 3 is pressed the next instruction will be skipped
        cpu.write_opcode_batch(&[(0x0202, 0xE19E)])?;
        // LD V2, 0x55 which should be skipped
        cpu.write_opcode_batch(&[(0x0204, 0x6255)])?;
        // LD V2, 0xAA which should execute after skip
        cpu.write_opcode_batch(&[(0x0206, 0x62AA), (0x0208, 0x1FFF)])?;
        cpu.run()?;
        assert_eq!(cpu.read_register(RegisterLabel::V2)?, 0xAA);
        Ok(())
    }

    #[test]
    fn sknp_operation() -> Result<(), Error> {
        // Scenario 1: Key not pressed so the next instruction is skipped
        {
            let mut cpu = Cpu::default();
            // LD V0, 4 so that V0 holds the key index to check
            cpu.write_opcode_batch(&[(0x0200, 0x6004)])?;
            // Ensure keyboard[4] is false
            cpu.keyboard[4] = false;
            // SKNP V0: if key 4 is not pressed, the next instruction will be skipped
            cpu.write_opcode_batch(&[(0x0202, 0xE0A1)])?;
            // LD V1, 0x55 which should be skipped
            cpu.write_opcode_batch(&[(0x0204, 0x6155)])?;
            // LD V1, 0xAA which should execute
            cpu.write_opcode_batch(&[(0x0206, 0x61AA), (0x0208, 0x1FFF)])?;
            cpu.run()?;
            assert_eq!(cpu.read_register(RegisterLabel::V1)?, 0xAA);
        }
        // Scenario 2: Key pressed so the next instruction is not skipped
        {
            let mut cpu = Cpu::default();
            // LD V0, 4 so that V0 holds the key index to check
            cpu.write_opcode_batch(&[(0x0200, 0x6004)])?;
            // Set keyboard[4] to true to simulate key press
            cpu.keyboard[4] = true;
            // SKNP V0: since key 4 is pressed the next instruction will not be skipped
            cpu.write_opcode_batch(&[(0x0202, 0xE0A1)])?;
            // LD V1, 0x55 which should execute
            cpu.write_opcode_batch(&[(0x0204, 0x6155), (0x0206, 0x1FFF)])?;
            cpu.run()?;
            assert_eq!(cpu.read_register(RegisterLabel::V1)?, 0x55);
        }
        Ok(())
    }

    #[test]
    fn ld_from_delay_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // Manually set delay timer to 77
        cpu.delay = 77;
        // LD from delay: Fx07 for V1 will load the delay timer into V1
        cpu.write_opcode_batch(&[(0x0200, 0xF107), (0x0202, 0x1FFF)])?;
        cpu.run()?;
        assert_eq!(cpu.read_register(RegisterLabel::V1)?, 77);
        Ok(())
    }

    #[test]
    fn ld_from_key_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // Set keyboard: simulate key 7 being pressed
        cpu.keyboard[7] = true;
        // LD from key: Fx0A for V1 will wait for a key press and load its value into V1
        cpu.write_opcode_batch(&[(0x0200, 0xF10A), (0x0202, 0x1FFF)])?;
        cpu.run()?;
        assert_eq!(cpu.read_register(RegisterLabel::V1)?, 7);
        Ok(())
    }

    #[test]
    fn ld_to_delay_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // LD V1, 0x21 to load 33 into V1
        cpu.write_opcode_batch(&[(0x0200, 0x6121)])?;
        // LD to delay: Fx15 for V1 stores V1 into the delay timer
        cpu.write_opcode_batch(&[(0x0202, 0xF115), (0x0204, 0x1FFF)])?;
        cpu.run()?;
        assert_eq!(cpu.delay, 0x21);
        Ok(())
    }

    #[test]
    fn ld_to_sound_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // LD V3, 0x2C to load 44 into V3
        cpu.write_opcode_batch(&[(0x0200, 0x632C)])?;
        // LD to sound: Fx18 for V3 stores V3 into the sound timer
        cpu.write_opcode_batch(&[(0x0202, 0xF318), (0x0204, 0x1FFF)])?;
        cpu.run()?;
        assert_eq!(cpu.sound, 0x2C);
        Ok(())
    }

    #[test]
    fn add_i_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // LD V0, 0x0A to load 10 into V0
        cpu.write_opcode_batch(&[(0x0200, 0x600A)])?;
        // ADD I, V0: F01E for V0 adds V0 to the index register
        cpu.write_opcode_batch(&[(0x0202, 0xF01E), (0x0204, 0x1FFF)])?;
        cpu.run()?;
        assert_eq!(cpu.index, 10);
        Ok(())
    }

    #[test]
    fn ld_i_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // LD V1, 7 to load 7 into V1
        cpu.write_opcode_batch(&[(0x0200, 0x6107)])?;
        // LD I: Fx29 for V1 sets I to 7 * 5
        cpu.write_opcode_batch(&[(0x0202, 0xF129), (0x0204, 0x1FFF)])?;
        cpu.run()?;
        assert_eq!(cpu.index, 7 * 5);
        Ok(())
    }

    #[test]
    fn reset_keyboard_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // Set some keys to true
        cpu.keyboard[2] = true;
        cpu.keyboard[5] = true;
        // Reset the keyboard
        cpu.reset_keyboard();
        for &key in cpu.keyboard.iter() {
            assert!(!key);
        }
        Ok(())
    }

    #[test]
    fn key_down_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // Simulate key 5 being pressed
        cpu.key_down(5);
        assert!(cpu.keyboard[5]);
        Ok(())
    }

    #[test]
    fn drw_operation_height_1() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        // Program:
        // 00E0: CLS             (clear the screen)
        // 6000: LD V0, 0        (set V0 = 0; x-coordinate)
        // 6100: LD V1, 0        (set V1 = 0; y-coordinate)
        // A300: LDI 0x300       (set index register I = 0x300)
        // D011: DRW V0,V1,1     (draw 1-byte sprite at (V0,V1))
        // 1FFF: JP 0xFFF       (jump to exit)
        let opcodes = [
            (0x0200, 0x00E0), // CLS
            (0x0202, 0x6000), // LD V0, 0
            (0x0204, 0x6100), // LD V1, 0
            (0x0206, 0xA300), // LDI 0x300
            (0x0208, 0xD011), // DRW V0,V1,1
            (0x020A, 0x1FFF), // JP 0xFFF (halt)
        ];
        cpu.write_opcode_batch(&opcodes)?;
        // Write sprite data into memory at 0x300:
        // Sprite: 0x81 -> 0b10000001, so pixel at column 0 and 7 are on
        cpu.write_memory(0x300, 0x81)?;

        cpu.run()?;

        // Verify that the sprite was drawn correctly
        // Check pixel at (0,0):
        assert_eq!(
            cpu.get_display_pixel(0u16),
            1,
            "Pixel at (0,0) should be on"
        );

        // Check pixel at (7,0):
        assert_eq!(
            cpu.get_display_pixel(7u16),
            1,
            "Pixel at (7,0) should be on"
        );

        // Verify pixels in between (columns 1 through 6) are off
        for col in 1..7 {
            let pixel_index = col as u16; // row 0, col = col
            assert_eq!(
                cpu.get_display_pixel(pixel_index),
                0,
                "Pixel at ({},0) should be off",
                col
            );
        }

        // The collision flag (VF) should remain 0
        assert_eq!(
            cpu.read_register(RegisterLabel::VF).unwrap(),
            0,
            "VF should be 0 for height 1 sprite"
        );
        Ok(())
    }

    #[test]
    fn drw_operation_height_3() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        // Program:
        // 00E0: CLS             (clear the screen)
        // 600A: LD V0, 10       (set V0 = 10; x-coordinate)
        // 6105: LD V1, 5        (set V1 = 5; y-coordinate)
        // A300: LDI 0x300       (set I = 0x300)
        // D013: DRW V0,V1,3     (draw 3-byte sprite at (V0,V1))
        // 1FFF: JP 0xFFF        (halt)
        let opcodes = [
            (0x0200, 0x00E0), // CLS
            (0x0202, 0x600A), // LD V0, 10
            (0x0204, 0x6105), // LD V1, 5
            (0x0206, 0xA300), // LDI 0x300
            (0x0208, 0xD013), // DRW V0,V1,3
            (0x020A, 0x1FFF), // JP 0xFFF (halt)
        ];
        cpu.write_opcode_batch(&opcodes)?;
        // Write sprite data into memory at 0x300:
        // Row 0: 0x3C -> 0b00111100
        // Row 1: 0x42 -> 0b01000010
        // Row 2: 0x81 -> 0b10000001
        cpu.write_memory(0x300, 0x3C)?;
        cpu.write_memory(0x301, 0x42)?;
        cpu.write_memory(0x302, 0x81)?;

        cpu.run()?;

        // Expected pattern for the drawn sprite (8 pixels per row)
        let expected = [
            [0, 0, 1, 1, 1, 1, 0, 0], // row 0 (0x3C)
            [0, 1, 0, 0, 0, 0, 1, 0], // row 1 (0x42)
            [1, 0, 0, 0, 0, 0, 0, 1], // row 2 (0x81)
        ];

        // Verify each pixel in the 3-row region drawn starting at (10,5)
        for row in 0..3 {
            for col in 0..8 {
                let pixel_x = (10 + col) % WIDTH;
                let pixel_y = (5 + row) % HEIGHT;
                let pixel_index = (pixel_y * WIDTH + pixel_x) as u16;
                assert_eq!(
                    cpu.get_display_pixel(pixel_index),
                    expected[row][col],
                    "Mismatch at row {} col {}",
                    row,
                    col
                );
            }
        }

        // Collision flag (VF) should be 0 since no pixels were erased
        assert_eq!(
            cpu.read_register(RegisterLabel::VF).unwrap(),
            0,
            "VF should be 0 for height 3 sprite"
        );
        Ok(())
    }
    #[test]
    fn drw_operation_edge_clipping() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        // Program:
        // 00E0: CLS              (clear the screen)
        // 6000: LD V0, 60       (set V0 = 60; x-coordinate is within display but near right edge)
        // 6105: LD V1, 5        (set V1 = 5; y-coordinate is within display)
        // A300: LDI 0x300       (set index register I = 0x300)
        // D011: DRW V0,V1,1     (draw 1-byte sprite at (V0,V1))
        // 1FFF: JP 0xFFF       (halt)
        let opcodes = [
            (0x0200, 0x00E0), // CLS
            (0x0202, 0x603C), // LD V0, 60    (0x6000 | 60 = 0x603C)
            (0x0204, 0x6105), // LD V1, 5     (0x6100 | 5  = 0x6105)
            (0x0206, 0xA300), // LDI 0x300
            (0x0208, 0xD011), // DRW V0,V1,1
            (0x020A, 0x1FFF), // JP 0xFFF (halt)
        ];
        cpu.write_opcode_batch(&opcodes)?;

        // Write sprite data: one byte sprite: 0xFF (all 8 pixels on)
        // When drawn at (60,5) on a 64-wide display, only pixels at x=60..63 should be drawn
        cpu.write_memory(0x300, 0xFF)?;

        cpu.run()?;

        // For row 5, check that only columns 60 to 63 are on, and all other columns are off
        for x in 0..WIDTH {
            let pixel_index = (5 * WIDTH + x) as u16;
            let pixel = cpu.get_display_pixel(pixel_index);
            if x >= 60 {
                assert_eq!(pixel, 1, "Pixel at ({},5) should be on", x);
            } else {
                assert_eq!(pixel, 0, "Pixel at ({},5) should be off", x);
            }
        }
        // Collision flag should be 0.
        assert_eq!(
            cpu.read_register(RegisterLabel::VF).unwrap(),
            0,
            "VF should be 0 when clipping without collision"
        );
        Ok(())
    }

    #[test]
    fn drw_operation_wrapping() -> Result<(), Error> {
        let mut cpu = Cpu::default();

        // Program:
        // 00E0: CLS              (clear the screen)
        // 6000: LD V0, 70       (set V0 = 70; completely off-screen horizontally)
        // 610A: LD V1, 10       (set V1 = 10; y-coordinate is in range)  (0x6100 | 10 = 0x610A)
        // A300: LDI 0x300       (set index register I = 0x300)
        // D011: DRW V0,V1,1     (draw 1-byte sprite at (V0,V1) – should wrap horizontally)
        // 1FFF: JP 0xFFF       (halt)
        let opcodes = [
            (0x0200, 0x00E0),      // CLS
            (0x0202, 0x6000 | 70), // LD V0, 70  (70 decimal)
            (0x0204, 0x6100 | 10), // LD V1, 10 (10 decimal)
            (0x0206, 0xA300),      // LDI 0x300
            (0x0208, 0xD011),      // DRW V0,V1,1
            (0x020A, 0x1FFF),      // JP 0xFFF (halt)
        ];
        cpu.write_opcode_batch(&opcodes)?;
        // Write sprite data: one byte sprite: 0xFF (all 8 pixels on).
        cpu.write_memory(0x300, 0xFF)?;

        cpu.run()?;

        // Since V0=70, wrapping yields 70 % 64 = 6.
        // For row 10, we expect columns 6..13 to be drawn (with no clipping since 13 < 64)
        for x in 6..14 {
            let pixel_index = (10 * WIDTH + x) as u16;
            let pixel = cpu.get_display_pixel(pixel_index);
            assert_eq!(pixel, 1, "Pixel at ({},10) should be on due to wrapping", x);
        }
        // Also check that a pixel outside the drawn region remains off
        let outside_index = (10 * WIDTH + 5) as u16;
        assert_eq!(
            cpu.get_display_pixel(outside_index),
            0,
            "Pixel at (5,10) should be off"
        );
        // Collision flag should be 0
        assert_eq!(
            cpu.read_register(RegisterLabel::VF).unwrap(),
            0,
            "VF should be 0 when wrapping without collision"
        );
        Ok(())
    }

    #[test]
    fn font_data_written() {
        let cpu = Cpu::default();
        let expected_font_data: [u8; 80] = FONT_DATA;

        let start = 0x00;
        let end = start + expected_font_data.len();
        assert_eq!(&cpu.memory[start..end], &expected_font_data);
    }
}
