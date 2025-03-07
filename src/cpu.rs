use crate::error::Error;
use crate::rom::Rom;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time;

// TODO Scrap most of the public API, tests can access props directly and it won't matter

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

// TODO Redo custom types/aliases

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
pub const CYCLE_INTERVAL: time::Duration = time::Duration::from_micros(1_000_000 / 700);

/// The interval of one sound and delay timer is 60hz
pub const SD_INTERVAL: time::Duration = time::Duration::from_micros(1_000_000 / 60);

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

    pub fn as_mut(&mut self) -> &mut [u8; 256] {
        &mut self.pixels
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
    sd_interval: time::Duration,
    cycle_interval: time::Duration,
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
            sd_interval: SD_INTERVAL,
            cycle_interval: CYCLE_INTERVAL,
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

        if (op_high_byte << 8 | op_low_byte) != 0 {
            println!("{}, {}", pc, self.memory[pc])
        }

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
    pub fn run_serial(&mut self) -> Result<(), Error> {
        let mut last_sd_update = time::Instant::now();
        let mut next_cycle_start = time::Instant::now() + self.cycle_interval;

        loop {
            if !self.step()? {
                break;
            }

            // Update timers if 1/60 sec has elapsed
            let now = time::Instant::now();
            if now.duration_since(last_sd_update) >= self.sd_interval {
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
            if now < next_cycle_start {
                thread::sleep(next_cycle_start - now);
            }

            next_cycle_start += self.cycle_interval;
        }

        Ok(())
    }

    /// Spawns a separate thread to run the CPU loop concurrently, handling cycle timing and timer
    /// updates
    pub fn run_concurrent(cpu: Arc<Mutex<Self>>) -> thread::JoinHandle<Result<(), Error>> {
        // All cycle information is defined here.
        // Lock the CPU briefly to extract the interval properties.
        let (cycle_interval, sd_interval) = {
            let cpu_ref = cpu.lock().expect("Failed to lock CPU");
            (cpu_ref.cycle_interval, cpu_ref.sd_interval)
        };

        thread::spawn(move || {
            let mut last_sd_update = time::Instant::now();
            let mut next_cycle_start: time::Instant = time::Instant::now() + cycle_interval;

            loop {
                {
                    // Lock the CPU only for this cycle
                    let mut cpu_lock = cpu.lock()?;

                    // Execute one instruction; if step returns false, exit
                    if !cpu_lock.step()? {
                        break;
                    }

                    // Update timers if the timer_interval has elapsed
                    let now = time::Instant::now();
                    if now.duration_since(last_sd_update) >= sd_interval {
                        if cpu_lock.delay > 0 {
                            cpu_lock.delay -= 1;
                        }
                        if cpu_lock.sound > 0 {
                            cpu_lock.sound -= 1;
                        }
                        last_sd_update = now;
                    }
                } // CPU lock is released here

                // Sleep until the next cycle
                let now = time::Instant::now();
                if now < next_cycle_start {
                    thread::sleep(next_cycle_start - now);
                }

                next_cycle_start += cycle_interval;
            }
            Ok(())
        })
    }

    /// Executes one CPU instruction and returns false if the program counter exceeds memory bounds
    pub fn step(&mut self) -> Result<bool, Error> {
        if self.program_counter >= 4095 {
            return Ok(false);
        }

        let opcode = self.fetch();
        let instruction = Self::decode(opcode);
        self.execute(instruction)?;

        Ok(true)
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
    fn ld_vx_into_vy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
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

    /// Returns a reference to the current value of the index register I
    pub fn read_index_register(&self) -> &AddressRegister {
        &self.index
    }

    /// Returns the value stored in the specified register if the index is within bounds
    pub fn read_register(&self, register_label: RegisterLabel) -> Result<VRegister, Error> {
        if let Some(value) = self.registers.get(register_label as usize).copied() {
            Ok(value)
        } else {
            Err(Error::Cpu("Out of bounds".to_string()))
        }
    }

    /// Writes a new value to the specified register after checking that the index is within bounds
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

    /// Returns the value stored at the specified memory address if the address is within bounds
    pub fn read_memory(&self, address: Address) -> Result<VRegister, Error> {
        if let Some(value) = self.memory.get(address as usize).copied() {
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

    /// Writes multiple values to memory using a list of (address, value) pairs
    pub fn write_memory_batch(
        &mut self,
        addresses_and_values: &[(Address, u8)],
    ) -> Result<(), Error> {
        for &(address, value) in addresses_and_values {
            self.write_memory(address, value)?;
        }
        Ok(())
    }

    /// Writes a 2-byte opcode to memory at the specified even address by splitting it into high and
    /// low bytes
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

    /// Writes multiple opcodes to memory using a list of (address, opcode) pairs
    pub fn write_opcode_batch(&mut self, opcodes: &[(Address, OpCode)]) -> Result<(), Error> {
        for &(address, opcode) in opcodes {
            self.write_opcode(address, opcode)?;
        }
        Ok(())
    }

    /// Grabs a rom and dumps the `OpCode`s from it into memory
    pub fn load_rom(&mut self, rom: Rom) -> Result<(), Error> {
        self.program_counter = 0x200;

        for (i, &byte) in rom.opcodes.iter().enumerate() {
            self.write_memory((self.program_counter + i) as u16, byte)?;
        }

        println!("{}, {}", &self.memory[0x200], &self.memory[0x201]);

        Ok(())
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

        cpu.run_serial()?;

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

        cpu.run_serial()?;

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

        cpu.run_serial()?;

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

        cpu.run_serial()?;

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

        cpu.run_serial()?;

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

        cpu.run_serial()?;

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
        cpu.run_serial()?;
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
        cpu.run_serial()?;
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
        cpu.run_serial()?;
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
        cpu.run_serial()?;
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
        cpu.run_serial()?;
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
        cpu.run_serial()?;
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
        cpu.run_serial()?;
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
        cpu.run_serial()?;
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
            cpu.run_serial()?;
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
            cpu.run_serial()?;
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
        for byte in cpu.display.as_mut().iter_mut() {
            *byte = 0xFF;
        }
        let opcodes = [
            (0x0200, 0x00E0), // CLS instruction
            (0x0202, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run_serial()?;
        // Verify that the entire display is cleared (all zeros)
        for &byte in cpu.display.as_slice().iter() {
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
        cpu.run_serial()?;
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
        // Expected: VF should be set to 0 and V0 becomes 5 - 10, which in wrapping arithmetic
        // yields 251
        let opcodes = [
            (0x0200, 0x8017), // 8xy7: for x = 0 and y = 1, perform V0 = V1 - V0
            (0x0202, 0x1FFF), // Jump to halt
        ];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run_serial()?;
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
            cpu.run_serial()?;
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
            cpu.run_serial()?;
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
            cpu.run_serial()?;
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
            cpu.run_serial()?;
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

        cpu.run_serial()?;

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

        cpu.run_serial()?;

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
        cpu.run_serial()?;

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
        cpu.run_serial()?;

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
        cpu.run_serial()?;

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
        // 0xC0FF: RND V0, 0xFF (generate a random number and AND with 0xFF, storing the result
        // in V0)
        // 0x1FFF: JP 0xFFF (halt execution)
        let opcodes = [(0x0200, 0xC0FF), (0x0202, 0x1FFF)];
        cpu.write_opcode_batch(&opcodes)?;
        cpu.run_serial()?;

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
        cpu.run_serial()?;
        assert_eq!(cpu.read_register(RegisterLabel::V1)?, 0xAA);
        Ok(())
    }

    #[test]
    fn skp_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // LD V1, 3 so that V1 holds the key index to check
        cpu.write_opcode_batch(&[(0x0200, 0x6103)])?;
        // Set keyboard[3] to true to simulate key press
        cpu.keyboard.key_down(3);
        // SKP V1: if key 3 is pressed the next instruction will be skipped
        cpu.write_opcode_batch(&[(0x0202, 0xE19E)])?;
        // LD V2, 0x55 which should be skipped
        cpu.write_opcode_batch(&[(0x0204, 0x6255)])?;
        // LD V2, 0xAA which should execute after skip
        cpu.write_opcode_batch(&[(0x0206, 0x62AA), (0x0208, 0x1FFF)])?;
        cpu.run_serial()?;
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
            cpu.keyboard.key_down(4);
            // SKNP V0: if key 4 is not pressed, the next instruction will be skipped
            cpu.write_opcode_batch(&[(0x0202, 0xE0A1)])?;
            // LD V1, 0x55 which should be skipped
            cpu.write_opcode_batch(&[(0x0204, 0x6155)])?;
            // LD V1, 0xAA which should execute
            cpu.write_opcode_batch(&[(0x0206, 0x61AA), (0x0208, 0x1FFF)])?;
            cpu.run_serial()?;
            assert_eq!(cpu.read_register(RegisterLabel::V1)?, 0xAA);
        }
        // Scenario 2: Key pressed so the next instruction is not skipped
        {
            let mut cpu = Cpu::default();
            // LD V0, 4 so that V0 holds the key index to check
            cpu.write_opcode_batch(&[(0x0200, 0x6004)])?;
            // Set keyboard[4] to true to simulate key press
            cpu.keyboard.key_down(4);
            // SKNP V0: since key 4 is pressed the next instruction will not be skipped
            cpu.write_opcode_batch(&[(0x0202, 0xE0A1)])?;
            // LD V1, 0x55 which should execute
            cpu.write_opcode_batch(&[(0x0204, 0x6155), (0x0206, 0x1FFF)])?;
            cpu.run_serial()?;
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
        cpu.run_serial()?;
        assert_eq!(cpu.read_register(RegisterLabel::V1)?, 77);
        Ok(())
    }

    #[test]
    fn ld_from_key_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // Set keyboard: simulate key 7 being pressed
        cpu.keyboard.key_down(7);
        // LD from key: Fx0A for V1 will wait for a key press and load its value into V1
        cpu.write_opcode_batch(&[(0x0200, 0xF10A), (0x0202, 0x1FFF)])?;
        cpu.run_serial()?;
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
        cpu.run_serial()?;
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
        cpu.run_serial()?;
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
        cpu.run_serial()?;
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
        cpu.run_serial()?;
        assert_eq!(cpu.index, 7 * 5);
        Ok(())
    }

    #[test]
    fn reset_keyboard_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // Set some keys to true
        cpu.keyboard.key_down(2);
        cpu.keyboard.key_down(5);
        // Reset the keyboard
        cpu.keyboard.clear();
        for &key in cpu.keyboard.as_slice().iter() {
            assert!(!key);
        }
        Ok(())
    }

    #[test]
    fn key_down_operation() -> Result<(), Error> {
        let mut cpu = Cpu::default();
        // Simulate key 5 being pressed
        cpu.keyboard.key_down(5);
        assert!(cpu.keyboard.as_slice()[5]);
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

        cpu.run_serial()?;

        // Verify that the sprite was drawn correctly
        // Check pixel at (0,0):
        assert_eq!(
            cpu.display.get_pixel(0u16),
            1,
            "Pixel at (0,0) should be on"
        );

        // Check pixel at (7,0):
        assert_eq!(
            cpu.display.get_pixel(7u16),
            1,
            "Pixel at (7,0) should be on"
        );

        // Verify pixels in between (columns 1 through 6) are off
        for col in 1..7 {
            let pixel_index = col as u16; // row 0, col = col
            assert_eq!(
                cpu.display.get_pixel(pixel_index),
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

        cpu.run_serial()?;

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
                    cpu.display.get_pixel(pixel_index),
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

        cpu.run_serial()?;

        // For row 5, check that only columns 60 to 63 are on, and all other columns are off
        for x in 0..WIDTH {
            let pixel_index = (5 * WIDTH + x) as u16;
            let pixel = cpu.display.get_pixel(pixel_index);
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

        cpu.run_serial()?;

        // Since V0=70, wrapping yields 70 % 64 = 6.
        // For row 10, we expect columns 6..13 to be drawn (with no clipping since 13 < 64)
        for x in 6..14 {
            let pixel_index = (10 * WIDTH + x) as u16;
            let pixel = cpu.display.get_pixel(pixel_index);
            assert_eq!(pixel, 1, "Pixel at ({},10) should be on due to wrapping", x);
        }
        // Also check that a pixel outside the drawn region remains off
        let outside_index = (10 * WIDTH + 5) as u16;
        assert_eq!(
            cpu.display.get_pixel(outside_index),
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
