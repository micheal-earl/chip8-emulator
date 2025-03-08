use crate::error::Error;
use crate::rom::Rom;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time;

// TODO Double check that properties that should be private actually are private
// -> cpu.display and cpu.keyboard can use getter/setters instead?
// TODO Use format!() macro for error outputs?

/// An OpCode is 16 bits (2 bytes). These bits determine what the cpu executes
type OpCode = u16;

/// Represents a decoded CHIP-8 `OpCode`, split into its component parts
///
/// For example, given an opcode `0x73EE`:
/// - The first byte (`0x73`) is the high byte and the second byte (`0xEE`) is the low byte
/// - The high nibble (first 4 bits) and the low nibble (last 4 bits) are extracted from each byte
///
/// The opcode is interpreted as follows:
///
/// | Field | Bits | Location                                | Description     |
/// |-------|------|-----------------------------------------|-----------------|
/// | `c`   | 4    | High byte, high nibble                  | Opcode group    |
/// | `x`   | 4    | High byte, low nibble                   | CPU register Vx |
/// | `y`   | 4    | Low byte, high nibble                   | CPU register Vy |
/// | `d`   | 4    | Low byte, low nibble                    | Opcode subgroup |
/// | `kk`  | 8    | Low byte, both nibbles                  | Immediate value |
/// | `nnn` | 12   | high byte's low nibble and the low byte | Memory address  |
pub struct Instruction {
    pub opcode_group: u8, // c
    pub register_x: u8,   // x
    pub register_y: u8,   // y
    pub opcode_minor: u8, // d
    pub integer_kk: u8,   // kk
    pub addr: Address,    // nnn
}

/// A VRegister is a memory location containing 8 bits
type VRegister = u8;

/// An AddressRegister is a memory location containing 16 bits
type AddressRegister = u16;

/// An address is a pointer value stored in an instruction
type Address = u16;

/// The CHIP-8 describes the 16 u8 registers using Vx where x is the register index
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

/// The interval of one Cpu cycle is 700hz
const CYCLE_INTERVAL: time::Duration = time::Duration::from_micros(1_000_000 / 700);

/// The interval of one sound and delay timer is 60hz
const SD_INTERVAL: time::Duration = time::Duration::from_micros(1_000_000 / 60);

/// Represents the CHIP-8 display buffer containing 2048 pixels stored as 256 bytes
pub struct DisplayBuffer {
    pixels: [u8; 256],
}

impl Default for DisplayBuffer {
    fn default() -> Self {
        DisplayBuffer { pixels: [0; 256] }
    }
}

impl DisplayBuffer {
    /// Retrieves the value of the pixel at the given index from the bit-packed display buffer
    pub fn get_pixel(&self, pixel_index: u16) -> u8 {
        let byte_index = (pixel_index / 8) as usize;
        let bit_index = 7 - (pixel_index % 8);
        (self.pixels[byte_index] >> bit_index) & 1
    }

    /// Writes a new value to the pixel at the specified index in the display buffer by setting or
    /// clearing the corresponding bit
    pub fn set_pixel(&mut self, pixel_index: u16, value: bool) -> Result<(), crate::error::Error> {
        let byte_index = (pixel_index / 8) as usize;
        let bit_index = 7 - (pixel_index % 8);
        if byte_index >= self.pixels.len() {
            return Err(crate::error::Error::Cpu(
                "Byte index out of bounds of display buffer".to_string(),
            ));
        }
        if value {
            self.pixels[byte_index] |= 1 << bit_index;
        } else {
            self.pixels[byte_index] &= !(1 << bit_index);
        }
        Ok(())
    }

    /// Returns a reference to the underlying pixel array
    pub fn as_slice(&self) -> &[u8; 256] {
        &self.pixels
    }

    /// Clears the display by turning all pixels off
    pub fn clear(&mut self) {
        self.pixels = [0; 256];
    }
}

/// Represents the state of the CHIP-8 keyboard with 16 keys
///
/// **Keyboard Layout:**
///
/// | Original CHIP-8 | Emulated CHIP-8 |
/// | --------------- | --------------- |
/// | 1 2 3 C         | 1 2 3 4         |
/// | 4 5 6 D         | Q W E R         |
/// | 7 8 9 E         | A S D F         |
/// | A 0 B F         | Z X C V         |
pub struct Keyboard {
    keys: [bool; 16],
}

impl Default for Keyboard {
    fn default() -> Self {
        Keyboard { keys: [false; 16] }
    }
}

impl Keyboard {
    /// Marks the specified key as pressed
    pub fn key_down(&mut self, key: u8) {
        if (key as usize) < self.keys.len() {
            self.keys[key as usize] = true;
        }
    }

    /// Returns a reference to the keys array representing the keyboard state
    pub fn as_slice(&self) -> &[bool; 16] {
        &self.keys
    }

    /// Resets the keyboard state by clearing all key presses
    pub fn clear(&mut self) {
        self.keys = [false; 16];
    }
}

/// Represents the state and behavior of a CHIP-8 virtual CPU
///
/// This structure emulates the hardware of a CHIP-8 system by maintaining:
/// - **Registers:** 16 8-bit registers used for computations
/// - **Memory:** A 4096-byte space that holds the program and data
/// - **Stack:** A 16-level call stack for subroutine management
/// - **Display:** A bit-packed buffer for the CHIP-8 graphics output
/// - **Keyboard:** An array tracking the state of 16 keys
/// - **Index Register:** A 16-bit register often used to store memory addresses
/// - **Stack Pointer:** The current top index of the call stack
/// - **Program Counter:** The address of the next instruction to execute
/// - **Delay Timer:** A timer that counts down at a fixed rate
/// - **Sound Timer:** A timer that counts down and triggers sound when non-zero
/// - **SD Interval:** The time interval for updating delay and sound timers
/// - **Cycle Interval:** The duration between consecutive CPU instruction cycles
/// - **RNG:** An 8-bit state for generating pseudo-random numbers
/// - **Lock:** A flag that pauses CPU execution until a key press is detected
pub struct Cpu {
    registers: [VRegister; 16],
    memory: [VRegister; 4096],
    stack: [AddressRegister; 16],
    pub display: DisplayBuffer,
    pub keyboard: Keyboard,
    index: AddressRegister,
    stack_pointer: usize,
    program_counter: usize,
    delay: VRegister,
    sound: VRegister,
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
            display: DisplayBuffer::default(),
            keyboard: Keyboard::default(),
            index: 0,
            stack_pointer: 0,
            program_counter: 0x0200,
            delay: 0,
            sound: 0,
            rng: seed,
            lock: false,
        };

        cpu.write_fonts_to_memory();

        cpu
    }
}

impl Cpu {
    /// Fetches the next OpCode from memory at the current program counter and increments the
    /// program counter if not locked
    fn fetch(&mut self) -> OpCode {
        let pc = self.program_counter;
        let op_high_byte = self.memory[pc] as u16;
        let op_low_byte = self.memory[pc + 1] as u16;

        if !self.lock {
            self.program_counter += 2;
        }

        op_high_byte << 8 | op_low_byte
    }

    /// Decodes a 16-bit `OpCode` into an `Instruction` by extracting its component fields
    fn decode(opcode: OpCode) -> Instruction {
        Instruction {
            opcode_group: ((opcode & 0xF000) >> 12) as u8, // c
            register_x: ((opcode & 0x0F00) >> 8) as u8,    // x
            register_y: ((opcode & 0x00F0) >> 4) as u8,    // y
            opcode_minor: (opcode & 0x000F) as u8,         // d
            integer_kk: (opcode & 0x00FF) as u8,           // kk
            addr: opcode & 0x0FFF,                         // nnn
        }
    }

    /// Executes a decoded `Instruction` by mapping its fields to the corresponding method
    fn execute(&mut self, decoded: Instruction) -> Result<(), Error> {
        let Instruction {
            opcode_group: c,
            register_x: x,
            register_y: y,
            opcode_minor: d,
            integer_kk: kk,
            addr: nnn,
        } = decoded;

        let vx = RegisterLabel::try_from(x)?;
        let vy = RegisterLabel::try_from(y)?;

        match (c, x, y, d) {
            (0x0, 0x0, 0x0, 0x0) => return Ok(()),
            (0x0, 0x0, 0xE, 0x0) => self.cls(),
            (0x0, 0x0, 0xE, 0xE) => self.ret(),
            (0x1, _, _, _) => self.jmp(nnn),
            (0x2, _, _, _) => self.call(nnn),
            (0x3, _, _, _) => self.se(vx, kk),
            (0x4, _, _, _) => self.sne(vx, kk),
            (0x5, _, _, 0x0) => self.se_xy(vx, vy),
            (0x6, _, _, _) => self.ld(vx, kk),
            (0x7, _, _, _) => self.add(vx, kk),
            (0x8, _, _, _) => match d {
                0x0 => self.ld_vy_into_vx(vx, vy),
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
                        "0x8 OpCode group with unknown subgroup".to_string(),
                    ))
                }
            },
            (0x9, _, _, 0x0) => self.sne_xy(vx, vy),
            (0xA, _, _, _) => self.ldi(nnn),
            (0xB, _, _, _) => self.jmp_x(nnn),
            (0xC, _, _, _) => self.rnd(vx, kk),
            (0xD, _, _, _) => self.drw(vx, vy, d)?,
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

    /// Runs the CPU in a sequential loop for headless operation, useful for unit tests
    #[allow(dead_code)]
    fn run_serial(&mut self) -> Result<(), Error> {
        let mut last_sd_update = time::Instant::now();
        let mut next_cycle_start = time::Instant::now() + CYCLE_INTERVAL;

        loop {
            // Do one cycle
            if !self.step()? {
                break;
            }

            self.decrement_timers(&mut last_sd_update);
            Self::wait_for_next_cycle(&mut next_cycle_start);
        }

        Ok(())
    }

    /// Spawns a separate thread to run the `Cpu` loop concurrently
    pub fn run_concurrent(cpu: Arc<Mutex<Self>>) -> thread::JoinHandle<Result<(), Error>> {
        thread::spawn(move || {
            let mut last_sd_update = time::Instant::now();
            let mut next_cycle_start = time::Instant::now() + CYCLE_INTERVAL;

            loop {
                // Lock *only* while stepping & updating timers
                {
                    let mut cpu_lock = cpu.lock()?;
                    if !cpu_lock.step()? {
                        break;
                    }
                    cpu_lock.decrement_timers(&mut last_sd_update);
                }

                Self::wait_for_next_cycle(&mut next_cycle_start);
            }

            Ok(())
        })
    }

    /// Executes one CPU instruction and returns false if the program counter exceeds memory bounds
    fn step(&mut self) -> Result<bool, Error> {
        if self.program_counter >= 4095 {
            return Ok(false);
        }

        let opcode = self.fetch();
        let instruction = Self::decode(opcode);
        self.execute(instruction)?;

        Ok(true)
    }

    /// Decrement the sound and delay timers based on the last time they were changed
    fn decrement_timers(&mut self, last_sd_update: &mut time::Instant) {
        // Check/update timers:
        let now = time::Instant::now();
        if now.duration_since(*last_sd_update) >= SD_INTERVAL {
            if self.delay > 0 {
                self.delay -= 1;
            }
            if self.sound > 0 {
                self.sound -= 1;
            }
            *last_sd_update = now;
        }
    }

    /// Wait for the next cycle of the `Cpu` to achieve desired clock rate
    fn wait_for_next_cycle(next_cycle_start: &mut time::Instant) {
        let now = time::Instant::now();
        if now < *next_cycle_start {
            thread::sleep(*next_cycle_start - now);
        }
        *next_cycle_start += CYCLE_INTERVAL;
    }

    /// 00E0 (CLS) - Clears the display by resetting the display buffer
    fn cls(&mut self) {
        self.display.clear();
    }

    /// 00EE (RET) - Returns from the current subroutine by updating the program counter from the
    /// stack
    fn ret(&mut self) {
        if self.stack_pointer == 0 {
            panic!("Stack Underflow!");
        }

        self.stack_pointer -= 1;
        let call_addr = self.stack[self.stack_pointer];
        self.program_counter = call_addr as usize;
    }

    /// 1nnn (JMP) - Sets the program counter to the specified address
    fn jmp(&mut self, addr: Address) {
        self.program_counter = addr as usize;
    }

    /// 2nnn (CALL) - Calls a subroutine at the specified address and pushes the current program
    /// counter onto the stack
    fn call(&mut self, addr: Address) {
        if self.stack_pointer >= self.stack.len() {
            panic!("Stack Overflow!")
        }

        self.stack[self.stack_pointer] = self.program_counter as u16;
        self.stack_pointer += 1;
        self.program_counter = addr as usize;
    }

    /// 3xkk (SE) - Skips the next instruction if the value in Vx equals the immediate value kk
    fn se(&mut self, vx: RegisterLabel, kk: u8) {
        if self.registers[vx as usize] == kk {
            self.program_counter += 2;
        }
    }

    /// 4xkk (SNE) - Skips the next instruction if the value in Vx does not equal the immediate
    /// value kk
    fn sne(&mut self, vx: RegisterLabel, kk: u8) {
        if self.registers[vx as usize] != kk {
            self.program_counter += 2;
        }
    }

    /// 5xy0 (SE) - Skips the next instruction if the values in Vx and Vy are equal
    fn se_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        if self.registers[vx as usize] == self.registers[vy as usize] {
            self.program_counter += 2;
        }
    }

    /// 6xkk (LD) - Loads the immediate value kk into register Vx
    fn ld(&mut self, vx: RegisterLabel, kk: u8) {
        self.registers[vx as usize] = kk;
    }

    /// 7xkk (ADD) - Adds the immediate value kk to register Vx with wrapping on overflow
    fn add(&mut self, vx: RegisterLabel, kk: u8) {
        // Use overflowing_add to ensure program does not panic
        (self.registers[vx as usize], _) = self.registers[vx as usize].overflowing_add(kk);
    }

    /// 8xy0 (LD) - Copies the value from register Vy into register Vx
    fn ld_vy_into_vx(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        self.registers[vx as usize] = self.registers[vy as usize];
    }

    /// 8xy1 (OR) - Performs a bitwise OR between registers Vx and Vy, storing the result in Vx
    fn or_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        self.registers[vx as usize] |= self.registers[vy as usize];
    }

    /// 8xy2 (AND) - Performs a bitwise AND between registers Vx and Vy, storing the result in Vx
    fn and_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        self.registers[vx as usize] &= self.registers[vy as usize];
    }

    /// 8xy3 (XOR) - Performs a bitwise XOR between registers Vx and Vy, storing the result in Vx
    fn xor_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        self.registers[vx as usize] ^= self.registers[vy as usize];
    }

    /// 8xy4 (ADD) - Adds the value in register Vy to register Vx, updating Vx and setting the carry
    /// flag in VF if overflow occurs
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

    /// 8xy5 (SUB) - Subtracts the value in register Vy from register Vx, storing the result in Vx
    /// and updating the carry flag in VF
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

    /// 8xy6 (SHR) - Shifts register Vx right by one bit and sets VF to the least-significant bit
    /// before the shift
    fn shr(&mut self, vx: RegisterLabel) {
        let vx_value = self.registers[vx as usize];
        self.registers[0xF] = vx_value & 1;
        self.registers[vx as usize] = vx_value >> 1;
    }

    /// 8xy7 (SUBN) - Subtracts register Vx from register Vy, stores the result in Vx,
    /// and sets VF based on carry
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

    /// 8xyE (SHL) - Shifts register Vx left by one bit and sets VF to the most-significant bit
    /// before the shift
    fn shl(&mut self, vx: RegisterLabel) {
        let vx_value = self.registers[vx as usize];
        self.registers[0xF] = (vx_value >> 7) & 1;
        self.registers[vx as usize] = vx_value << 1;
    }

    /// 9xy0 (SNE) - Skips the next instruction if the values in registers Vx and Vy are not equal
    fn sne_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        if self.registers[vx as usize] != self.registers[vy as usize] {
            self.program_counter += 2;
        }
    }

    /// Annn (LDI) - Loads the address nnn into the index register I
    fn ldi(&mut self, nnn: u16) {
        self.index = nnn;
    }

    /// Bnnn (JP) - Jumps to the address nnn plus the value in register V0
    fn jmp_x(&mut self, nnn: u16) {
        self.program_counter = nnn as usize + (self.registers[0] as usize);
    }

    /// xkk (RND) - Generates a random number (0 to 255), ANDs it with kk, and stores the result in
    /// register Vx
    fn rnd(&mut self, vx: RegisterLabel, kk: u8) {
        // Update the RNG state using a simple 8-bit linear congruential generator
        // Using a multiplier of 37 and an increment of 1 which yields a full period mod 256
        self.rng = self.rng.wrapping_mul(37).wrapping_add(1);

        // Use the new state as the random value (0..=255)
        let random_value = self.rng;

        // Store the result of random_value AND kk in Vx
        self.registers[vx as usize] = random_value & kk;
    }

    /// Dxyn (DRW) - Draws a sprite of height d from memory location I at coordinates (Vx, Vy)
    ///
    /// The sprite is XORed onto the display and VF is set if any pixel is erased
    ///
    /// If the sprite is partially off-screen, it is clipped (pixels outside are not drawn)  
    ///
    /// If the sprite is completely off-screen, it wraps around  
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

                    let current_pixel = self.display.get_pixel(pixel_index);
                    let new_pixel = current_pixel ^ 1;
                    if current_pixel == 1 && new_pixel == 0 {
                        self.registers[0xF] = 1;
                    }
                    self.display.set_pixel(pixel_index, new_pixel == 1)?;
                }
            }
        }

        Ok(())
    }

    /// Ex9E (SKP) - Skips the next instruction if the key corresponding to the value in Vx is
    /// pressed
    fn skp(&mut self, vx: RegisterLabel) {
        if self.keyboard.as_slice()[self.registers[vx as usize] as usize] {
            self.program_counter += 2;
        }
    }

    /// ExA1 (SKNP) - Skips the next instruction if the key corresponding to the value in Vx is not
    /// pressed
    fn sknp(&mut self, vx: RegisterLabel) {
        if !self.keyboard.as_slice()[self.registers[vx as usize] as usize] {
            self.program_counter += 2;
        }
    }

    /// Fx07 (LD) - Loads the current delay timer value into register Vx
    fn ld_from_delay(&mut self, vx: RegisterLabel) {
        self.registers[vx as usize] = self.delay;
    }

    /// Fx0A (LD) - Waits for a key press and then stores the key value into register Vx, locking
    /// the CPU until input is received
    fn ld_from_key(&mut self, vx: RegisterLabel) {
        self.lock = true;
        let keys = self.keyboard.as_slice();
        for i in 0..keys.len() {
            if keys[i] {
                self.registers[vx as usize] = i as u8;
                self.lock = false;
                break;
            }
        }
    }

    /// Fx15 (LD) - Sets the delay timer to the value in register Vx
    fn ld_to_delay(&mut self, vx: RegisterLabel) {
        self.delay = self.registers[vx as usize];
    }

    /// Fx18 (LD) - Sets the sound timer to the value in register Vx
    fn ld_to_sound(&mut self, vx: RegisterLabel) {
        self.sound = self.registers[vx as usize];
    }

    ///  Fx1E (ADD I) - Adds the value in register Vx to the index register I
    fn add_i(&mut self, vx: RegisterLabel) {
        self.index += self.registers[vx as usize] as u16;
    }

    /// Fx29 (LD F) - Sets the index register I to the location of the sprite for the hexadecimal
    /// digit in Vx
    fn ld_i(&mut self, vx: RegisterLabel) {
        let digit = self.registers[vx as usize];
        self.index = (digit as u16) * 5;
    }

    /// Fx33 (LD B) - Stores the binary-coded decimal representation of Vx in memory at addresses I,
    /// I+1, and I+2
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

    /// Fx55 (LD) - Stores registers V0 through Vx into memory starting at the address in the index
    /// register I
    fn ld_from_registers(&mut self, vx: RegisterLabel) {
        let mut index = self.index;
        let stop = vx as u8;
        for i in 0..=stop {
            self.memory[index as usize] = self.registers[i as usize];
            index += 1;
        }
    }

    /// Fx65 (LD) - Loads registers V0 through Vx from memory starting at the address in the index
    /// register I
    fn ld_to_registers(&mut self, vx: RegisterLabel) {
        let mut index = self.index;
        let stop = vx as u8;
        for i in 0..=stop {
            self.registers[i as usize] = self.memory[index as usize];
            index += 1
        }
    }

    /// Writes the CHIP-8 font data into memory starting at address 0x000
    fn write_fonts_to_memory(&mut self) {
        // The font data occupies 80 bytes starting at memory address 0x000.
        let start = 0x000;
        let end = start + FONT_DATA.len();
        self.memory[start..end].copy_from_slice(&FONT_DATA);
    }

    /// Returns the value stored in the specified register if the index is within bounds
    pub fn read_register(&self, register_label: RegisterLabel) -> Result<VRegister, Error> {
        if let Some(value) = self.registers.get(register_label as usize).copied() {
            Ok(value)
        } else {
            Err(Error::Cpu("Out of bounds".to_string()))
        }
    }

    /// Writes a new value to the specified memory address after verifying that the address is
    /// within bounds
    pub fn write_memory(&mut self, address: Address, value: u8) -> Result<(), Error> {
        let index = address as usize;
        if index < self.memory.len() {
            self.memory[index] = value;
            Ok(())
        } else {
            Err(Error::Cpu("Memory address out of bounds".to_string()))
        }
    }

    /// Returns a reference to the current value of the sound timer
    pub fn read_sound(&self) -> &u8 {
        &self.sound
    }

    /// Grabs a rom and dumps the `OpCode`s from it into memory
    pub fn load_rom(&mut self, rom: Rom) -> Result<(), Error> {
        self.program_counter = 0x200;

        for (i, &byte) in rom.opcodes.iter().enumerate() {
            self.write_memory((self.program_counter + i) as u16, byte)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests for logical operations
    mod logic {
        use super::*;

        /// 8xy1 (OR)
        #[test]
        fn or_xy_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.registers[0x0] = 0xA; // 0b1010
            cpu.registers[0x1] = 0x5; // 0b0101

            // 0x0200 -> 0x80, 0x11 (V0 = V0 OR V1)
            // 0x0202 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0x80;
            cpu.memory[0x0201] = 0x11;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;

            // 0b1010 OR 0b0101 = 0b1111 (0xF)
            assert_eq!(cpu.registers[0x0], 0xF);

            Ok(())
        }

        /// 8xy2 (AND)
        #[test]
        fn and_xy_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.registers[0x0] = 0xA; // 0b1010
            cpu.registers[0x1] = 0x6; // 0b0110

            // 0x0200 -> 0x80, 0x12 (V0 = V0 AND V1)
            // 0x0202 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0x80;
            cpu.memory[0x0201] = 0x12;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;
            // 0b1010 AND 0b0110 = 0b0010 (2)
            assert_eq!(cpu.registers[0x0], 0x2);

            Ok(())
        }

        /// 8xy3 (XOR)
        #[test]
        fn xor_xy_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.registers[0x0] = 0xA; // 0b1010
            cpu.registers[0x1] = 0x6; // 0b0110

            // 0x0200 -> 0x80, 0x13 (V0 = V0 XOR V1)
            // 0x0202 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0x80;
            cpu.memory[0x0201] = 0x13;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;
            // 0b1010 XOR 0b0110 = 0b1100 (0xC)
            assert_eq!(cpu.registers[0x0], 0xC);

            Ok(())
        }
    }

    /// Tests for control flow
    mod control {
        use super::*;

        /// 1nnn (JMP)
        #[test]
        fn jump_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.registers[0x0] = 5;
            cpu.registers[0x1] = 7;

            // 0x0200 -> 0x13, 0x00  (Jump to 0x300)
            // 0x0300 -> 0x80, 0x14  (ADD V1 to V0)
            // 0x0302 -> 0x1F, 0xFF  (Jump to 0xFFF -> halt)
            cpu.memory[0x0200] = 0x13;
            cpu.memory[0x0201] = 0x00;
            cpu.memory[0x0300] = 0x80;
            cpu.memory[0x0301] = 0x14;
            cpu.memory[0x0302] = 0x1F;
            cpu.memory[0x0303] = 0xFF;

            cpu.run_serial()?;

            // V0 should now be 12
            assert_eq!(cpu.registers[0x0], 12);

            Ok(())
        }

        /// 2nnn (CALL) and 00EE (RET)
        #[test]
        fn call_and_ret_operations() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // V0=5, V1=10
            cpu.registers[0x0] = 5;
            cpu.registers[0x1] = 10;

            // 0x0200 -> 0x23, 0x00 (CALL 0x300)
            // 0x0202 -> 0x23, 0x00 (CALL 0x300 again)
            // 0x0204 -> 0x1F, 0xFF (JUMP 0xFFF -> halt)
            // 0x0300 -> 0x80, 0x14 (ADD V1 to V0)
            // 0x0302 -> 0x80, 0x14 (ADD V1 to V0 again)
            // 0x0304 -> 0x00, 0xEE (RET)
            cpu.memory[0x0200] = 0x23;
            cpu.memory[0x0201] = 0x00;
            cpu.memory[0x0202] = 0x23;
            cpu.memory[0x0203] = 0x00;
            cpu.memory[0x0204] = 0x1F;
            cpu.memory[0x0205] = 0xFF;

            cpu.memory[0x0300] = 0x80;
            cpu.memory[0x0301] = 0x14;
            cpu.memory[0x0302] = 0x80;
            cpu.memory[0x0303] = 0x14;
            cpu.memory[0x0304] = 0x00;
            cpu.memory[0x0305] = 0xEE;

            cpu.run_serial()?;

            // V0 was 5, then +10 twice per call, called twice => 5 + 10 + 10 + 10 + 10 = 45
            assert_eq!(cpu.registers[0x0], 45);

            Ok(())
        }

        /// 3xkk (SE)
        #[test]
        fn se_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.registers[0x0] = 10;

            // 0x0200 -> 0x30, 0x0A (SE V0, 0x0A => skip next if V0 == 10)
            // 0x0202 -> 0x60, 0x05 (LD V0, 0x05) <-- should be skipped
            // 0x0204 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0x30;
            cpu.memory[0x0201] = 0x0A;
            cpu.memory[0x0202] = 0x60;
            cpu.memory[0x0203] = 0x05;
            cpu.memory[0x0204] = 0x1F;
            cpu.memory[0x0205] = 0xFF;

            cpu.run_serial()?;

            // Should still be 10, as the LD was skipped
            assert_eq!(cpu.registers[0x0], 10);

            Ok(())
        }

        /// 4xkk (SNE)
        #[test]
        fn sne_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.registers[0x0] = 10;

            // 0x0200 -> 0x40, 0x05 (SNE V0, 0x05 => skip next if V0 != 5)
            // 0x0202 -> 0x60, 0x03 (LD V0, 0x03) <-- should be skipped
            // 0x0204 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0x40;
            cpu.memory[0x0201] = 0x05;
            cpu.memory[0x0202] = 0x60;
            cpu.memory[0x0203] = 0x03;
            cpu.memory[0x0204] = 0x1F;
            cpu.memory[0x0205] = 0xFF;

            cpu.run_serial()?;

            // Should still be 10
            assert_eq!(cpu.registers[0x0], 10);

            Ok(())
        }

        /// 5xy0 (SE Vx, Vy)
        #[test]
        fn se_xy_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.registers[0x0] = 20;
            cpu.registers[0x1] = 20;

            // 0x0200 -> 0x50, 0x10 (SE V0, V1 => skip next if V0 == V1)
            // 0x0202 -> 0x60, 0x05 (LD V0, 0x05) <-- should be skipped
            // 0x0204 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0x50;
            cpu.memory[0x0201] = 0x10;
            cpu.memory[0x0202] = 0x60;
            cpu.memory[0x0203] = 0x05;
            cpu.memory[0x0204] = 0x1F;
            cpu.memory[0x0205] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.registers[0x0], 20);

            Ok(())
        }

        /// 9xy0 (SNE Vx, Vy)
        #[test]
        fn sne_xy_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.registers[0x0] = 15;
            cpu.registers[0x1] = 20;

            // 0x0200 -> 0x90, 0x10 (SNE V0, V1 => skip next if V0 != V1)
            // 0x0202 -> 0x60, 0x99 (LD V0, 0x99) <-- should be skipped
            // 0x0204 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0x90;
            cpu.memory[0x0201] = 0x10;
            cpu.memory[0x0202] = 0x60;
            cpu.memory[0x0203] = 0x99;
            cpu.memory[0x0204] = 0x1F;
            cpu.memory[0x0205] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.registers[0x0], 15);

            Ok(())
        }

        /// Bnnn (JP V0)
        #[test]
        fn jmp_x_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.registers[0x0] = 5;

            // 0x0200 -> 0x60, 0x05 (LD V0, 5)  already set above, but let's double-check
            cpu.memory[0x0200] = 0x60;
            cpu.memory[0x0201] = 0x05;

            // 0x0202 -> 0xB2, 0x0B (JP 0x20B + V0 => 0x210)
            cpu.memory[0x0202] = 0xB2;
            cpu.memory[0x0203] = 0x0B;

            // At 0x0210: 0x61, 0xAA => LD V1, 0xAA
            // 0x0212 -> 0x1F, 0xFF  => halt
            cpu.memory[0x0210] = 0x61;
            cpu.memory[0x0211] = 0xAA;
            cpu.memory[0x0212] = 0x1F;
            cpu.memory[0x0213] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.registers[0x1], 0xAA);

            Ok(())
        }
    }

    /// Tests for math operations
    mod math {
        use super::*;

        /// 8xy4 (ADD) - Tests adding Vy to Vx over multiple instructions
        #[test]
        fn add_xy_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // Directly set register values
            cpu.registers[0x0] = 5;
            cpu.registers[0x1] = 10;
            cpu.registers[0x2] = 15;
            cpu.registers[0x3] = 7;

            // Write opcodes to memory
            // 0x0200 -> 0x80, 0x14  (ADD V1 to V0)
            // 0x0202 -> 0x80, 0x24  (ADD V2 to V0)
            // 0x0204 -> 0x80, 0x34  (ADD V3 to V0)
            // 0x0206 -> 0x1F, 0xFF  (Jump to 0xFFF - halt)
            cpu.memory[0x0200] = 0x80;
            cpu.memory[0x0201] = 0x14;
            cpu.memory[0x0202] = 0x80;
            cpu.memory[0x0203] = 0x24;
            cpu.memory[0x0204] = 0x80;
            cpu.memory[0x0205] = 0x34;
            cpu.memory[0x0206] = 0x1F;
            cpu.memory[0x0207] = 0xFF;

            cpu.run_serial()?;

            // Check final value in V0 (should be 5+10+15+7 = 37)
            assert_eq!(cpu.registers[0x0], 37);

            Ok(())
        }

        /// 7xkk (ADD) - Tests adding an immediate to Vx
        #[test]
        fn add_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // Set initial value in V1
            cpu.registers[0x1] = 20;

            // 0x0200 -> 0x71, 0x05  (Add 0x05 to V1)
            // 0x0202 -> 0x1F, 0xFF  (Jump to 0xFFF to halt)
            cpu.memory[0x0200] = 0x71;
            cpu.memory[0x0201] = 0x05;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;

            // V1 should be 25
            assert_eq!(cpu.registers[0x1], 25);

            Ok(())
        }

        /// 8xy5 (SUB)
        #[test]
        fn sub_xy_operation() -> Result<(), Error> {
            // Case 1: V0 > V1
            {
                let mut cpu = Cpu::default();
                cpu.registers[0x0] = 10;
                cpu.registers[0x1] = 5;

                // 0x0200 -> 0x80, 0x15 (V0 = V0 - V1)
                // 0x0202 -> 0x1F, 0xFF (halt)
                cpu.memory[0x0200] = 0x80;
                cpu.memory[0x0201] = 0x15;
                cpu.memory[0x0202] = 0x1F;
                cpu.memory[0x0203] = 0xFF;

                cpu.run_serial()?;
                // V0 = 5, VF=1
                assert_eq!(cpu.registers[0x0], 5);
                assert_eq!(cpu.registers[0xF], 1);
            }
            // Case 2: V0 <= V1
            {
                let mut cpu = Cpu::default();
                cpu.registers[0x0] = 5;
                cpu.registers[0x1] = 10;

                cpu.memory[0x0200] = 0x80;
                cpu.memory[0x0201] = 0x15;
                cpu.memory[0x0202] = 0x1F;
                cpu.memory[0x0203] = 0xFF;

                cpu.run_serial()?;
                // 5 - 10 => 251 (with wrapping), VF=0
                assert_eq!(cpu.registers[0x0], 251);
                assert_eq!(cpu.registers[0xF], 0);
            }

            Ok(())
        }

        /// 8xy7 (SUBN)
        #[test]
        fn subn_xy_operation_no_borrow() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            // V1=10 > V0=5 => V0 = 10 - 5 => 5, VF=1
            cpu.registers[0x0] = 5;
            cpu.registers[0x1] = 10;

            // 0x0200 -> 0x80, 0x17
            // 0x0202 -> 0x1F, 0xFF
            cpu.memory[0x0200] = 0x80;
            cpu.memory[0x0201] = 0x17;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.registers[0xF], 1);
            assert_eq!(cpu.registers[0x0], 5);
            Ok(())
        }

        #[test]
        fn subn_xy_operation_with_borrow() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            // V1=5 <= V0=10 => V0 = 5 - 10 => 0xFB, VF=0
            cpu.registers[0x0] = 10;
            cpu.registers[0x1] = 5;

            cpu.memory[0x0200] = 0x80;
            cpu.memory[0x0201] = 0x17;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.registers[0xF], 0);
            assert_eq!(cpu.registers[0x0], 251);
            Ok(())
        }

        /// 8xyE (SHL)
        #[test]
        fn shl_operation() -> Result<(), Error> {
            // 1) V0=0x90 => shift left => 0x20, VF=1
            {
                let mut cpu = Cpu::default();
                cpu.registers[0x0] = 0x90;

                // 0x0200 -> 0x80, 0x0E
                // 0x0202 -> 0x1F, 0xFF
                cpu.memory[0x0200] = 0x80;
                cpu.memory[0x0201] = 0x0E;
                cpu.memory[0x0202] = 0x1F;
                cpu.memory[0x0203] = 0xFF;

                cpu.run_serial()?;
                assert_eq!(cpu.registers[0x0], 0x20);
                assert_eq!(cpu.registers[0xF], 1);
            }
            // 2) V1=0x30 => shift left => 0x60, VF=0
            {
                let mut cpu = Cpu::default();
                cpu.registers[0x1] = 0x30;

                // 0x0200 -> 0x81, 0x1E
                // 0x0202 -> 0x1F, 0xFF
                cpu.memory[0x0200] = 0x81;
                cpu.memory[0x0201] = 0x1E;
                cpu.memory[0x0202] = 0x1F;
                cpu.memory[0x0203] = 0xFF;

                cpu.run_serial()?;
                assert_eq!(cpu.registers[0x1], 0x60);
                assert_eq!(cpu.registers[0xF], 0);
            }
            Ok(())
        }

        /// 8xy6 (SHR)
        #[test]
        fn shr_operation() -> Result<(), Error> {
            // 1) V0=5 => shift right => 2, VF=1
            {
                let mut cpu = Cpu::default();
                cpu.registers[0x0] = 5;

                cpu.memory[0x0200] = 0x80;
                cpu.memory[0x0201] = 0x06;
                cpu.memory[0x0202] = 0x1F;
                cpu.memory[0x0203] = 0xFF;

                cpu.run_serial()?;
                assert_eq!(cpu.registers[0x0], 2);
                assert_eq!(cpu.registers[0xF], 1);
            }
            // 2) V1=4 => shift right => 2, VF=0
            {
                let mut cpu = Cpu::default();
                cpu.registers[0x1] = 4;

                cpu.memory[0x0200] = 0x81;
                cpu.memory[0x0201] = 0x06;
                cpu.memory[0x0202] = 0x1F;
                cpu.memory[0x0203] = 0xFF;

                cpu.run_serial()?;
                assert_eq!(cpu.registers[0x1], 2);
                assert_eq!(cpu.registers[0xF], 0);
            }
            Ok(())
        }

        /// Cxkk (RND)
        #[test]
        fn test_rnd_opcode() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            // Force a known RNG seed
            cpu.rng = 123;

            // 0x0200 -> 0xC0, 0xFF (RND V0, 0xFF)
            // 0x0202 -> 0x1F, 0xFF
            cpu.memory[0x0200] = 0xC0;
            cpu.memory[0x0201] = 0xFF;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;

            // After the LCG update, rng= (123*37 +1) & 0xFF = 0xC8 => 200
            // So V0 should be 200
            assert_eq!(cpu.registers[0x0], 200);

            Ok(())
        }

        /// Fx1E (ADD I, Vx)
        #[test]
        fn add_i_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // V0=10, I=0 by default
            cpu.memory[0x0200] = 0x60;
            cpu.memory[0x0201] = 0x0A; // LD V0, 10
                                       // 0x0202 -> 0xF0, 0x1E => I += V0
                                       // 0x0204 -> 0x1F, 0xFF
            cpu.memory[0x0202] = 0xF0;
            cpu.memory[0x0203] = 0x1E;
            cpu.memory[0x0204] = 0x1F;
            cpu.memory[0x0205] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.index, 10);

            Ok(())
        }
    }

    /// Tests for loading operations
    mod loads {
        use super::*;

        /// 6xkk (LD)
        #[test]
        fn ld_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // 0x0200 -> 0x63, 0xAB (LD V3, 0xAB)
            // 0x0202 -> 0x1F, 0xFF (Jump to 0xFFF -> halt)
            cpu.memory[0x0200] = 0x63;
            cpu.memory[0x0201] = 0xAB;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;

            assert_eq!(cpu.registers[0x3], 0xAB);

            Ok(())
        }

        /// Annn (LDI)
        #[test]
        fn ldi_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // 0x0200 -> 0xA1, 0x23 (LDI 0x123)
            // 0x0202 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0xA1;
            cpu.memory[0x0201] = 0x23;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;

            assert_eq!(cpu.index, 0x0123);

            Ok(())
        }

        /// 8xy0 (LD Vx, Vy)
        #[test]
        fn ld_vx_into_vy_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.registers[0x1] = 42;

            // 0x0200 -> 0x80, 0x10 (LD V0, V1)
            // 0x0202 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0x80;
            cpu.memory[0x0201] = 0x10;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.registers[0x0], 42);

            Ok(())
        }

        /// Fx55 (LD [I], V0..Vx)
        #[test]
        fn ld_from_registers_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // Set V0..V5 to known values
            cpu.registers[0] = 0;
            cpu.registers[1] = 2;
            cpu.registers[2] = 4;
            cpu.registers[3] = 6;
            cpu.registers[4] = 8;
            cpu.registers[5] = 10;

            // 0x0200 -> 0xA0, 0xC8 (I=0x0C8 -> 200 dec)
            // 0x0202 -> 0xF5, 0x55 (LD [I], V0..V5)
            // 0x0204 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0xA0;
            cpu.memory[0x0201] = 0xC8;
            cpu.memory[0x0202] = 0xF5;
            cpu.memory[0x0203] = 0x55;
            cpu.memory[0x0204] = 0x1F;
            cpu.memory[0x0205] = 0xFF;

            cpu.run_serial()?;

            // Check memory from 200..206
            for i in 0..=5 {
                assert_eq!(cpu.memory[200 + i], cpu.registers[i]);
            }
            // Next memory cell should be 0
            assert_eq!(cpu.memory[206], 0);

            Ok(())
        }

        /// Fx65 (LD V0..Vx, [I])
        #[test]
        fn ld_to_registers_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // Preload memory at 150..155 => 11,22,33,44,55,66
            cpu.memory[150] = 11;
            cpu.memory[151] = 22;
            cpu.memory[152] = 33;
            cpu.memory[153] = 44;
            cpu.memory[154] = 55;
            cpu.memory[155] = 66;

            // 0x0200 -> 0xA0, 0x96 (I=150 decimal)
            // 0x0202 -> 0xF5, 0x65 (LD V0..V5, [I])
            // 0x0204 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0xA0;
            cpu.memory[0x0201] = 0x96;
            cpu.memory[0x0202] = 0xF5;
            cpu.memory[0x0203] = 0x65;
            cpu.memory[0x0204] = 0x1F;
            cpu.memory[0x0205] = 0xFF;

            cpu.run_serial()?;

            assert_eq!(cpu.registers[0], 11);
            assert_eq!(cpu.registers[1], 22);
            assert_eq!(cpu.registers[2], 33);
            assert_eq!(cpu.registers[3], 44);
            assert_eq!(cpu.registers[4], 55);
            assert_eq!(cpu.registers[5], 66);

            // V6 remains 0
            assert_eq!(cpu.registers[6], 0);

            Ok(())
        }

        /// Fx33 (LD BCD)
        #[test]
        fn ld_bcd_operation() -> Result<(), Error> {
            // Example: V1=123 => memory[I..I+2] => 1,2,3
            {
                let mut cpu = Cpu::default();
                // 0x0200 -> 0x61, 0x7B (LD V1, 123)
                // 0x0202 -> 0xA1, 0x2C (LDI 0x12C => 300)
                // 0x0204 -> 0xF1, 0x33 (Fx33 => store BCD of V1)
                // 0x0206 -> 0x1F, 0xFF (halt)
                cpu.memory[0x0200] = 0x61;
                cpu.memory[0x0201] = 0x7B;
                cpu.memory[0x0202] = 0xA1;
                cpu.memory[0x0203] = 0x2C;
                cpu.memory[0x0204] = 0xF1;
                cpu.memory[0x0205] = 0x33;
                cpu.memory[0x0206] = 0x1F;
                cpu.memory[0x0207] = 0xFF;

                cpu.run_serial()?;
                assert_eq!(cpu.memory[300], 1);
                assert_eq!(cpu.memory[301], 2);
                assert_eq!(cpu.memory[302], 3);
            }
            // Example: V2=0 => ...
            {
                let mut cpu = Cpu::default();
                // 0x0200 -> 0x62, 0x00
                // 0x0202 -> 0xA1, 0x36  (310)
                // 0x0204 -> 0xF2, 0x33
                // 0x0206 -> 0x1F, 0xFF
                cpu.memory[0x0200] = 0x62;
                cpu.memory[0x0201] = 0x00;
                cpu.memory[0x0202] = 0xA1;
                cpu.memory[0x0203] = 0x36;
                cpu.memory[0x0204] = 0xF2;
                cpu.memory[0x0205] = 0x33;
                cpu.memory[0x0206] = 0x1F;
                cpu.memory[0x0207] = 0xFF;

                cpu.run_serial()?;
                assert_eq!(cpu.memory[310], 0);
                assert_eq!(cpu.memory[311], 0);
                assert_eq!(cpu.memory[312], 0);
            }
            // Example: V3=255 => 255 => digits 2,5,5
            {
                let mut cpu = Cpu::default();
                cpu.memory[0x0200] = 0x63;
                cpu.memory[0x0201] = 0xFF;
                cpu.memory[0x0202] = 0xA1;
                cpu.memory[0x0203] = 0x44; // 324
                cpu.memory[0x0204] = 0xF3;
                cpu.memory[0x0205] = 0x33;
                cpu.memory[0x0206] = 0x1F;
                cpu.memory[0x0207] = 0xFF;

                cpu.run_serial()?;
                assert_eq!(cpu.memory[324], 2);
                assert_eq!(cpu.memory[325], 5);
                assert_eq!(cpu.memory[326], 5);
            }
            Ok(())
        }

        /// Fx07 (LD Vx, delay)
        #[test]
        fn ld_from_delay_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.delay = 77;

            // 0x0200 -> 0xF1, 0x07 => V1 = delay
            // 0x0202 -> 0x1F, 0xFF => halt
            cpu.memory[0x0200] = 0xF1;
            cpu.memory[0x0201] = 0x07;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.registers[0x1], 77);

            Ok(())
        }

        /// Fx15 (LD delay, Vx)
        #[test]
        fn ld_to_delay_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            // 0x0200 -> 0x61, 0x21 => V1=0x21
            // 0x0202 -> 0xF1, 0x15 => delay=V1
            // 0x0204 -> 0x1F, 0xFF
            cpu.memory[0x0200] = 0x61;
            cpu.memory[0x0201] = 0x21;
            cpu.memory[0x0202] = 0xF1;
            cpu.memory[0x0203] = 0x15;
            cpu.memory[0x0204] = 0x1F;
            cpu.memory[0x0205] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.delay, 0x21);

            Ok(())
        }

        /// Fx29 (LD F, Vx)
        #[test]
        fn ld_i_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            // V1=7 => I=7*5=35
            cpu.memory[0x0200] = 0x61;
            cpu.memory[0x0201] = 0x07;
            // 0x0202 -> 0xF1, 0x29
            // 0x0204 -> 0x1F, 0xFF
            cpu.memory[0x0202] = 0xF1;
            cpu.memory[0x0203] = 0x29;
            cpu.memory[0x0204] = 0x1F;
            cpu.memory[0x0205] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.index, 35);
            Ok(())
        }

        /// Fx18 (LD sound, Vx)
        #[test]
        fn ld_to_sound_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // 0x0200 -> 0x63, 0x2C => V3=0x2C
            // 0x0202 -> 0xF3, 0x18 => sound=V3
            // 0x0204 -> 0x1F, 0xFF
            cpu.memory[0x0200] = 0x63;
            cpu.memory[0x0201] = 0x2C;
            cpu.memory[0x0202] = 0xF3;
            cpu.memory[0x0203] = 0x18;
            cpu.memory[0x0204] = 0x1F;
            cpu.memory[0x0205] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.sound, 0x2C);

            Ok(())
        }
    }

    /// Tests for keyboard operations
    mod keyboard {
        use super::*;
        /// Basic keyboard usage
        #[test]
        fn reset_keyboard_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.keyboard.key_down(2);
            cpu.keyboard.key_down(5);
            cpu.keyboard.clear();

            for &pressed in cpu.keyboard.as_slice().iter() {
                assert!(!pressed);
            }
            Ok(())
        }

        #[test]
        fn key_down_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            cpu.keyboard.key_down(5);
            assert!(cpu.keyboard.as_slice()[5]);
            Ok(())
        }

        /// Fx0A (LD Vx, K)
        #[test]
        fn ld_from_key_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            // Press key[7]
            cpu.keyboard.key_down(7);

            // 0x0200 -> 0xF1, 0x0A (Wait for key, store in V1)
            // 0x0202 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0xF1;
            cpu.memory[0x0201] = 0x0A;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;
            assert_eq!(cpu.registers[0x1], 7);

            Ok(())
        }

        /// Ex9E (SKP)
        #[test]
        fn skp_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();
            // V1=3 => we'll check for key 3
            cpu.memory[0x0200] = 0x61;
            cpu.memory[0x0201] = 0x03;
            // 0x0202 -> 0xE1, 0x9E => SKP V1 (skip next if key[3] pressed)
            // 0x0204 -> 0x62, 0x55 => LD V2, 0x55 (should be skipped)
            // 0x0206 -> 0x62, 0xAA => LD V2, 0xAA (executed if skip worked)
            // 0x0208 -> 0x1F, 0xFF => halt
            cpu.memory[0x0202] = 0xE1;
            cpu.memory[0x0203] = 0x9E;
            cpu.memory[0x0204] = 0x62;
            cpu.memory[0x0205] = 0x55;
            cpu.memory[0x0206] = 0x62;
            cpu.memory[0x0207] = 0xAA;
            cpu.memory[0x0208] = 0x1F;
            cpu.memory[0x0209] = 0xFF;

            // Mark key[3] pressed
            cpu.keyboard.key_down(3);

            cpu.run_serial()?;

            // We expect V2=0xAA (the 0x55 load was skipped)
            assert_eq!(cpu.registers[0x2], 0xAA);
            Ok(())
        }

        /// ExA1 (SKNP)
        #[test]
        fn sknp_operation() -> Result<(), Error> {
            // Scenario 1: Key not pressed => skip next
            {
                let mut cpu = Cpu::default();
                // V0=4 => we'll test key[4]
                cpu.memory[0x0200] = 0x60;
                cpu.memory[0x0201] = 0x04;
                // 0x0202 -> 0xE0, 0xA1 => SKNP V0 (skip next if key[4] is not pressed)
                // 0x0204 -> 0x61, 0x55 => LD V1, 0x55 (skipped)
                // 0x0206 -> 0x61, 0xAA => LD V1, 0xAA (executed if skip is triggered)
                // 0x0208 -> 0x1F, 0xFF => halt
                cpu.memory[0x0202] = 0xE0;
                cpu.memory[0x0203] = 0xA1;
                cpu.memory[0x0204] = 0x61;
                cpu.memory[0x0205] = 0x55;
                cpu.memory[0x0206] = 0x61;
                cpu.memory[0x0207] = 0xAA;
                cpu.memory[0x0208] = 0x1F;
                cpu.memory[0x0209] = 0xFF;

                // We do NOT press key[4]
                cpu.run_serial()?;
                // Should have skipped LD V1,0x55 => so V1=0xAA
                assert_eq!(cpu.registers[0x1], 0xAA);
            }

            // Scenario 2: Key pressed => do not skip
            {
                let mut cpu = Cpu::default();
                cpu.memory[0x0200] = 0x60;
                cpu.memory[0x0201] = 0x04;
                cpu.memory[0x0202] = 0xE0;
                cpu.memory[0x0203] = 0xA1;
                cpu.memory[0x0204] = 0x61;
                cpu.memory[0x0205] = 0x55;
                cpu.memory[0x0206] = 0x1F;
                cpu.memory[0x0207] = 0xFF;

                // Press key[4]
                cpu.keyboard.key_down(4);

                cpu.run_serial()?;
                // Should NOT skip LD => V1=0x55
                assert_eq!(cpu.registers[0x1], 0x55);
            }
            Ok(())
        }
    }

    /// Tests for display operations
    mod display {
        use super::*;

        /// 00E0 (CLS)
        #[test]
        fn cls_operation() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // Turn on a few pixels using set_pixel
            cpu.display.set_pixel(0, true)?;
            cpu.display.set_pixel(10, true)?;
            cpu.display.set_pixel(100, true)?;
            cpu.display.set_pixel(150, true)?;

            // 0x0200 -> 0x00, 0xE0 (CLS)
            // 0x0202 -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0x00;
            cpu.memory[0x0201] = 0xE0;
            cpu.memory[0x0202] = 0x1F;
            cpu.memory[0x0203] = 0xFF;

            cpu.run_serial()?;

            // All display bytes should be 0
            for &byte in cpu.display.as_slice() {
                assert_eq!(byte, 0);
            }
            Ok(())
        }

        /// Dxyn (DRW) - height=1
        #[test]
        fn drw_operation_height_1() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // 0x0200 -> 0x00, 0xE0 (CLS)
            // 0x0202 -> 0x60, 0x00 (V0=0)
            // 0x0204 -> 0x61, 0x00 (V1=0)
            // 0x0206 -> 0xA3, 0x00 (I=0x300)
            // 0x0208 -> 0xD0, 0x11 (DRW V0, V1, 1)
            // 0x020A -> 0x1F, 0xFF (halt)
            cpu.memory[0x0200] = 0x00;
            cpu.memory[0x0201] = 0xE0;
            cpu.memory[0x0202] = 0x60;
            cpu.memory[0x0203] = 0x00;
            cpu.memory[0x0204] = 0x61;
            cpu.memory[0x0205] = 0x00;
            cpu.memory[0x0206] = 0xA3;
            cpu.memory[0x0207] = 0x00;
            cpu.memory[0x0208] = 0xD0;
            cpu.memory[0x0209] = 0x11;
            cpu.memory[0x020A] = 0x1F;
            cpu.memory[0x020B] = 0xFF;

            // Write sprite data (1 byte) at 0x300
            // 0x81 => 1000 0001 => pixels on at col 0 and col 7
            cpu.memory[0x300] = 0x81;

            cpu.run_serial()?;

            // pixel(0,0) => on, pixel(7,0) => on, others off
            let top_left = cpu.display.get_pixel(0);
            let top_right = cpu.display.get_pixel(7);
            assert_eq!(top_left, 1);
            assert_eq!(top_right, 1);
            for col in 1..7 {
                assert_eq!(cpu.display.get_pixel(col as u16), 0);
            }
            assert_eq!(cpu.registers[0xF], 0);

            Ok(())
        }

        /// Dxyn (DRW) - height=3
        #[test]
        fn drw_operation_height_3() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // 0x0200 -> 0x00, 0xE0 (CLS)
            // 0x0202 -> 0x60, 0x0A (V0=10)
            // 0x0204 -> 0x61, 0x05 (V1=5)
            // 0x0206 -> 0xA3, 0x00 (I=0x300)
            // 0x0208 -> 0xD0, 0x13 (DRW V0,V1,3)
            // 0x020A -> 0x1F, 0xFF
            cpu.memory[0x0200] = 0x00;
            cpu.memory[0x0201] = 0xE0;
            cpu.memory[0x0202] = 0x60;
            cpu.memory[0x0203] = 0x0A;
            cpu.memory[0x0204] = 0x61;
            cpu.memory[0x0205] = 0x05;
            cpu.memory[0x0206] = 0xA3;
            cpu.memory[0x0207] = 0x00;
            cpu.memory[0x0208] = 0xD0;
            cpu.memory[0x0209] = 0x13;
            cpu.memory[0x020A] = 0x1F;
            cpu.memory[0x020B] = 0xFF;

            // Sprite data
            cpu.memory[0x300] = 0x3C; // 0011 1100
            cpu.memory[0x301] = 0x42; // 0100 0010
            cpu.memory[0x302] = 0x81; // 1000 0001

            cpu.run_serial()?;

            // Just check collision flag
            assert_eq!(cpu.registers[0xF], 0);

            // Spot-check a couple of pixels
            // The first row is at (10,5) => bits 00111100 => columns 10+2..10+5 on
            // For thoroughness you could assert them all
            Ok(())
        }

        #[test]
        fn drw_operation_edge_clipping() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // Program:
            // 00E0: CLS
            // 603C: LD V0, 60 (x=60 near the right edge on a 64-wide display)
            // 6105: LD V1, 5
            // A300: LDI 0x300
            // D011: DRW V0, V1, 1  (Draw 1 row)
            // 1FFF: JP 0xFFF (halt)
            cpu.memory[0x0200] = 0x00;
            cpu.memory[0x0201] = 0xE0;
            cpu.memory[0x0202] = 0x60;
            cpu.memory[0x0203] = 0x3C; // decimal 60
            cpu.memory[0x0204] = 0x61;
            cpu.memory[0x0205] = 0x05;
            cpu.memory[0x0206] = 0xA3;
            cpu.memory[0x0207] = 0x00;
            cpu.memory[0x0208] = 0xD0;
            cpu.memory[0x0209] = 0x11;
            cpu.memory[0x020A] = 0x1F;
            cpu.memory[0x020B] = 0xFF;

            // Write one-byte sprite data of 0xFF => 1111 1111
            // That means 8 pixels wide
            // Starting at x=60 should clip columns 60..63 and discard columns 64..67 off-screen
            cpu.memory[0x300] = 0xFF;

            cpu.run_serial()?;

            // Verify that columns 60..63 in row 5 are set, while columns <60 or >=64 are off
            for x in 0..64 {
                let pixel_index = (5 * 64 + x) as u16;
                let pixel_val = cpu.display.get_pixel(pixel_index);
                if x >= 60 && x < 64 {
                    assert_eq!(pixel_val, 1, "Pixel at x={} should be on", x);
                } else {
                    assert_eq!(pixel_val, 0, "Pixel at x={} should be off", x);
                }
            }

            // No collision if we only turned pixels on, so VF = 0
            assert_eq!(cpu.registers[0xF], 0);

            Ok(())
        }

        #[test]
        fn drw_operation_wrapping() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // Program:
            // 00E0: CLS
            // 6046: LD V0, 70 (completely off-screen to the right => 70 % 64 = 6)
            // 610A: LD V1, 10
            // A300: LDI 0x300
            // D011: DRW V0, V1, 1
            // 1FFF: JP 0xFFF
            cpu.memory[0x0200] = 0x00;
            cpu.memory[0x0201] = 0xE0;
            cpu.memory[0x0202] = 0x60;
            cpu.memory[0x0203] = 0x46; // 70 decimal
            cpu.memory[0x0204] = 0x61;
            cpu.memory[0x0205] = 0x0A; // 10 decimal
            cpu.memory[0x0206] = 0xA3;
            cpu.memory[0x0207] = 0x00;
            cpu.memory[0x0208] = 0xD0;
            cpu.memory[0x0209] = 0x11;
            cpu.memory[0x020A] = 0x1F;
            cpu.memory[0x020B] = 0xFF;

            // Sprite data 0xFF => columns 0..7 if no clipping
            // But x=70 means we wrap => effectively draws at x=6..13
            cpu.memory[0x300] = 0xFF;

            cpu.run_serial()?;

            // Check row 10, columns 6..13 should be on, everything else off
            for x in 0..64 {
                let pixel_index = (10 * 64 + x) as u16;
                let val = cpu.display.get_pixel(pixel_index);
                if x >= 6 && x < 14 {
                    assert_eq!(val, 1, "Expected pixel (x={},y=10) to be on", x);
                } else {
                    assert_eq!(val, 0, "Expected pixel (x={},y=10) to be off", x);
                }
            }
            // VF=0 => no collision
            assert_eq!(cpu.registers[0xF], 0);

            Ok(())
        }
    }

    /// Tests for bounding errors
    mod bounds {
        use super::*;

        /// Attempt to call more times than stack size => should panic with "Stack Overflow!"
        #[test]
        #[should_panic(expected = "Stack Overflow!")]
        fn stack_overflow() {
            let mut cpu = Cpu::default();

            // We want 17 consecutive CALL instructions, each calling the *next* instruction address
            // For a 16-element stack, calling 17 times should exceed stack size and trigger the
            // panic

            // Starting at 0x200, place call instructions at 0x200, 0x202, 0x204, ...
            // so that each one calls the next slot. No RET instructions, so the stack never pops
            for i in 0..17 {
                // Address of the current CALL
                let addr = 0x200 + i * 2;
                // The target is the next 2-byte slot
                let target = 0x200 + (i + 1) * 2;

                // 2nnn => CALL nnn, so we split `target` into high/low bytes
                // The high nibble is 0x2, then the lower 12 bits are `target`
                // For example, if target = 0x206 => 0x2 0x06 => 0x2206 in big-endian form
                // top nibble '2', plus the high nibble of target
                let high_byte = 0x20 | ((target >> 8) as u8 & 0x0F);
                let low_byte = (target & 0xFF) as u8;

                cpu.memory[addr] = high_byte;
                cpu.memory[addr + 1] = low_byte;
            }

            // Now run the CPU. By the time it processes the 17th call instruction,
            // stack_pointer >= 16 => panic("Stack Overflow!")
            cpu.run_serial().unwrap();
        }

        /// Attempt to return when stack pointer is zero => "Stack Underflow!"
        #[test]
        #[should_panic(expected = "Stack Underflow!")]
        fn stack_underflow() {
            let mut cpu = Cpu::default();
            // 00EE => RET
            cpu.memory[0x200] = 0x00;
            cpu.memory[0x201] = 0xEE;
            cpu.run_serial().unwrap();
        }

        /// Check if the program counter exceeds its bounds
        #[test]
        fn out_of_bounds_program_counter() -> Result<(), Error> {
            let mut cpu = Cpu::default();

            // Place the program counter near the end of RAM: 4094 is last valid full-instruction
            // boundary
            cpu.program_counter = 4094;
            // The next fetch would try to read memory[4094] and memory[4095].
            // If we do one more step after that, we'd go beyond 4095 => step() should return false
            // (or do nothing further)

            // We can just run a few steps. The CPU’s logic says if program_counter >= 4095 =>
            // return Ok(false)

            // Should succeed reading from 4094..4095 (whatever is in there)
            let step1 = cpu.step()?;

            // Should come back false because PC is now beyond 4095
            let step2 = cpu.step()?;

            // The first step might read uninitialized memory, but it won't panic,
            // it just interprets it as some opcode. The second step should say "we're done."
            assert!(step1, "First step should still 'work' at the edge");
            assert!(
                !step2,
                "Second step after crossing 0xFFF boundary should return false"
            );
            Ok(())
        }
    }

    /// Tests for general checks
    mod checks {
        use super::*;

        /// Verify that font data was written at init
        #[test]
        fn font_data_written() {
            let cpu = Cpu::default();
            // Confirm that memory[0..80] == FONT_DATA
            for (i, &font_byte) in FONT_DATA.iter().enumerate() {
                assert_eq!(cpu.memory[i], font_byte);
            }
        }
    }
}
