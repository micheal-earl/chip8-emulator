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

// TODO Replace with proper error
/// Placeholder type for errors
pub type CpuError = &'static str;

/// The CHIP-8 describes the 16 u8 registers using Vx where x is the register index.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
    type Error = CpuError;

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
            _ => Err("Invalid register value"),
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

pub const DURATION_700HZ_IN_MICROS: time::Duration = time::Duration::from_micros(1428);
pub const DURATION_1HZ_IN_MICROS: time::Duration = time::Duration::from_micros(1_000_000);

pub struct Cpu {
    registers: [Register; 16],
    memory: [Register; 4096],
    stack: [Register16; 16],
    display: Display, //[[u8; 64]; 32],
    delay: Register,  // TODO delay timer
    sound: Register,  // TODO sound timer
    index: Register16,
    stack_pointer: usize,
    program_counter: usize,
}

impl Default for Cpu {
    fn default() -> Self {
        let mut cpu = Self {
            registers: [0; 16],
            memory: [0; 4096],
            stack: [0; 16],
            display: [0; 256],
            delay: 0,
            sound: 0,
            index: 0,
            stack_pointer: 0,
            program_counter: 0x0200,
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
            println!(
                "{}, {}, {}",
                self.display[0], self.display[5], self.display[10]
            )
        }
        // DEBUG ^

        self.program_counter += 2;

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

    fn execute(&mut self, decoded: Instruction) -> Result<(), CpuError> {
        let (c, x, y, d, kk, nnn) = decoded;
        match (c, x, y, d) {
            (0x0, 0x0, 0x0, 0x0) => return Ok(()), // 0000
            (0x0, 0x0, 0xE, 0x0) => self.cls(),    // 00E0 Clear the screen
            (0x0, 0x0, 0xE, 0xE) => self.ret(),    // 00EE Return from subroutine
            (0x1, _, _, _) => self.jmp(nnn),       // 1nnn Jump to location nnn
            (0x2, _, _, _) => self.call(nnn),      // Call subroutine at location nnn
            (0x3, _, _, _) => self.se(RegisterLabel::try_from(x)?, kk), // 3xkk Skip next instruction if Vx == kk
            (0x4, _, _, _) => self.sne(RegisterLabel::try_from(x)?, kk), // 4xkk Skip next instruvtion if Vx != kk
            (0x5, _, _, 0x0) => {
                self.se_xy(RegisterLabel::try_from(x)?, RegisterLabel::try_from(y)?)
                // 5xy0 Skip next instruction if Vx == Vy
            }
            (0x6, _, _, _) => self.ld(RegisterLabel::try_from(x)?, kk), // 6xkk Write kk to Vx
            (0x7, _, _, _) => self.add(RegisterLabel::try_from(x)?, kk), // 7xkk Add kk to Vx, write result to Vx
            (0x8, _, _, _) => match d {
                0x0 => todo!("8xy0 LD Vx, Vy"),
                0x1 => todo!("8xy1 OR Vx, Vy"),
                0x2 => todo!("8xy2 AND Vx, Vy"),
                0x3 => todo!("8xy3 XOR Vx, Vy"),
                0x4 => self.add_xy(RegisterLabel::try_from(x)?, RegisterLabel::try_from(y)?),
                0x5 => todo!("8xy5 SUB Vx, Vy"),
                0x6 => todo!("8xy6 SHR Vx, Vy"),
                0x7 => todo!("8xy7 SUBN Vx, Vy"),
                0xE => todo!("8xyE SHL Vx , Vy"),
                _ => return Err("0x8 instruction with unknown d value"),
            },
            (0x9, _, _, 0x0) => {
                self.sne_xy(RegisterLabel::try_from(x)?, RegisterLabel::try_from(y)?)
                // 9xy0 Skip next instruction if Vx != Vy
            }
            (0xA, _, _, _) => self.ldi(nnn),
            (0xB, _, _, _) => todo!("Bnnn JP V0, addr"),
            (0xC, _, _, _) => todo!("Cxkk RND Vx, byte"),
            (0xD, _, _, _) => {
                self.drw(RegisterLabel::try_from(x)?, RegisterLabel::try_from(y)?, d)?
                // Dxyn Display n-byte sprite starting at memory location I at (Vx, Vy), set VF = collision.
            }
            // TODO Use nested match statements like the 0x8 insturctions?
            (0xE, _, 0x9, 0xE) => todo!("Ex9E SKP - Skip next instruction if key Vx is pressed"),
            (0xE, _, 0xA, 0x1) => {
                todo!("ExA1 SKNP - Skip next instruction if key Vx is not pressed")
            }
            (0xF, _, 0x0, 0x7) => {
                todo!("Fx07 LD Vx, delay - value of delay timer is placed into Vx")
            }
            (0xF, _, 0x0, 0xA) => {
                todo!("Fx0A LD Vx, delay - value of delay timer is placed into Vx")
            }
            (0xF, _, 0x1, 0x5) => todo!("Fx15 LD delay, Vx - delay is set to Vx"),
            (0xF, _, 0x1, 0x8) => todo!("Fx18 LD sound, Vx - sound is set to Vx"),
            (0xF, _, 0x1, 0xE) => todo!("Fx1E LD delay, Vx - delay is set to Vx"),
            (0xF, _, 0x2, 0x9) => todo!("Fx29 LD F, Vx"),
            (0xF, _, 0x3, 0x3) => todo!("Fx33"),
            (0xF, _, 0x5, 0x5) => todo!("Fx55"),
            (0xF, _, 0x6, 0x5) => todo!("Fx65"),
            _ => return Err("Unknown instruction"),
        }

        Ok(())
    }

    /// Runs the cpu, stepping through instructions until end of memory
    pub fn run(&mut self) -> Result<(), CpuError> {
        // TODO: Add 60hz timer for sound and delay
        let interval = DURATION_700HZ_IN_MICROS;
        let mut time_after_interval = time::Instant::now() + interval;

        loop {
            let s = self.step()?;
            if !s {
                break;
            }

            // Sleep until the next cycle
            let now = time::Instant::now();
            if now < time_after_interval {
                thread::sleep(time_after_interval - now);
            }

            time_after_interval += interval;
        }

        Ok(())
    }

    /// Steps the cpu forward one instruction
    pub fn step(&mut self) -> Result<bool, CpuError> {
        if self.program_counter >= 4095 {
            return Ok(false); // or return false if you want to exit
        }

        let opcode = self.fetch();
        let instruction = Self::decode(opcode);
        self.execute(instruction)?;

        Ok(true)
    }

    /// (00E0) CLEAR the display
    fn cls(&mut self) {
        self.display = [0; 256];
    }

    /// (1nnn) JUMP to `addr`
    fn jmp(&mut self, addr: Address) {
        self.program_counter = addr as usize;
    }

    /// (3xkk) SE Skip next instruction if Vx == kk
    fn se(&mut self, vx: RegisterLabel, kk: u8) {
        if self.registers[vx as usize] == kk {
            self.program_counter += 2;
        }
    }

    /// (4xkk) SNE Skip if Vx != kk
    fn sne(&mut self, vx: RegisterLabel, kk: u8) {
        if self.registers[vx as usize] != kk {
            self.program_counter += 2;
        }
    }

    /// (5xy0) SE Skip next instruction if Vx == Vy
    fn sne_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        if self.registers[vx as usize] != self.registers[vy as usize] {
            self.program_counter += 2;
        }
    }

    /// (6xy0) SE Skip next instruction if Vx == Vy
    fn se_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        if self.registers[vx as usize] == self.registers[vy as usize] {
            self.program_counter += 2;
        }
    }

    /// (6xkk) LD sets the value `kk` into register `Vx`
    fn ld(&mut self, vx: RegisterLabel, kk: u8) {
        self.registers[vx as usize] = kk;
    }

    /// (Annn) LDI sets the value `nnn` into the index register (I register)
    fn ldi(&mut self, nnn: u16) {
        self.index = nnn;
    }

    /// (7xkk) ADD adds the value `kk` to register `Vx` and stores the sum in 'Vx'
    fn add(&mut self, vx: RegisterLabel, kk: u8) {
        // Use overflowing_add to ensure program does not panic
        (self.registers[vx as usize], _) = self.registers[vx as usize].overflowing_add(kk);
    }

    /// (2nnn) CALL sub-routine at `addr`
    fn call(&mut self, addr: Address) {
        if self.stack_pointer >= self.stack.len() {
            panic!("Stack Overflow!")
        }

        self.stack[self.stack_pointer] = self.program_counter as u16;
        self.stack_pointer += 1;
        self.program_counter = addr as usize;
    }

    /// (00ee) RET return from the current sub-routine
    fn ret(&mut self) {
        if self.stack_pointer == 0 {
            panic!("Stack Underflow!");
        }

        self.stack_pointer -= 1;
        let call_addr = self.stack[self.stack_pointer];
        self.program_counter = call_addr as usize;
    }

    // (7xkk) Add one registers contents to another registers contents
    fn add_xy(&mut self, vx: RegisterLabel, vy: RegisterLabel) {
        let arg1 = self.registers[vx as usize];
        let arg2 = self.registers[vy as usize];

        let (val, overflow) = arg1.overflowing_add(arg2);
        self.registers[vx as usize] = val;

        if overflow {
            self.registers[0xF] = 1;
        } else {
            self.registers[0xF] = 0;
        }
    }

    /// Display d-byte sprite starting at memory location I at (Vx, Vy)  
    /// Sprites are XOR’d onto the display  
    /// If the sprite is partially off-screen, it is clipped (pixels outside are not drawn)  
    /// If the sprite is completely off-screen, it wraps around  
    /// VF is set to 1 if any drawn pixel is erased  
    fn drw(&mut self, vx: RegisterLabel, vy: RegisterLabel, d: u8) -> Result<(), CpuError> {
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

    /// Helper to get the value of an individual pixel from the bit-packed display  
    /// The display is stored as 256 u8’s, each holding 8 pixels  
    /// `pixel` is the overall pixel index (0..2047)  
    fn get_display_pixel(&self, pixel_index: u16) -> u8 {
        let byte_index = (pixel_index / 8) as usize;
        let bit_index = 7 - (pixel_index % 8);
        (self.display[byte_index] >> bit_index) & 1
    }

    pub fn write_display(&mut self, pixel_index: u16, value: bool) -> Result<(), CpuError> {
        // Calculate which byte holds the pixel.
        let byte_index = (pixel_index / 8) as usize;
        // Calculate the bit position within that byte.
        // We assume bit 7 is the leftmost pixel, so we subtract the remainder from 7.
        let bit_index = 7 - (pixel_index % 8);

        // Check if the byte index is valid for our display buffer.
        if byte_index >= self.display.len() {
            return Err("Byte index out of bounds of display buffer.");
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
    pub fn read_register(&self, register_label: RegisterLabel) -> Result<Register, CpuError> {
        if let Some(value) = self.registers.get(register_label as usize).copied() {
            Ok(value)
        } else {
            Err("Out of bounds")
        }
    }

    /// Write a new value to Vx
    pub fn write_register(
        &mut self,
        register_label: RegisterLabel,
        value: u8,
    ) -> Result<(), CpuError> {
        let register_label_as_usize = register_label as usize;
        let register_is_in_bounds = register_label_as_usize < self.registers.len();

        if register_is_in_bounds {
            self.registers[register_label_as_usize] = value;
            Ok(())
        } else {
            Err("Register index out of bounds")
        }
    }

    /// Returns a copy of the value stored in the provided memory address
    pub fn read_memory(&self, address: Address) -> Result<Register, CpuError> {
        if let Some(value) = self.memory.get(address as usize).copied() {
            Ok(value)
        } else {
            Err("Out of bounds")
        }
    }

    /// Write a new value to a memory location
    pub fn write_memory(&mut self, address: Address, value: u8) -> Result<(), CpuError> {
        let index = address as usize;
        if index < self.memory.len() {
            self.memory[index] = value;
            Ok(())
        } else {
            Err("Memory address out of bounds")
        }
    }

    /// Write multiple new values to several memory locations
    pub fn write_memory_batch(
        &mut self,
        addresses_and_values: &[(Address, u8)],
    ) -> Result<(), CpuError> {
        for &(address, value) in addresses_and_values {
            self.write_memory(address, value)?;
        }
        Ok(())
    }

    /// Write an opcode to memory. Opcodes are 2 registers long
    pub fn write_opcode(&mut self, address: Address, opcode: OpCode) -> Result<(), CpuError> {
        let index = address as usize;
        if index % 2 != 0 {
            return Err(
                "Cannot insert opcode at odd memory address. Opcodes are 2 registers long.",
            );
        }
        if index + 1 >= self.memory.len() {
            return Err("Memory address out of bounds");
        }
        // Split the instruction into high and low bytes.
        self.memory[index] = (opcode >> 8) as u8;
        self.memory[index + 1] = (opcode & 0xFF) as u8;
        Ok(())
    }

    pub fn write_opcode_batch(&mut self, opcodes: &[(Address, OpCode)]) -> Result<(), CpuError> {
        for &(address, opcode) in opcodes {
            self.write_opcode(address, opcode)?;
        }
        Ok(())
    }

    /// Write multiple opcodes to memory at several memory locations
    fn write_fonts_to_memory(&mut self) {
        // The font data occupies 80 bytes starting at memory address 0x000.
        // We don't use write_memory_batch here as that would be slower
        let start = 0x000;
        let end = start + FONT_DATA.len();
        self.memory[start..end].copy_from_slice(&FONT_DATA);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_xy_operation() -> Result<(), CpuError> {
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
    fn add_operation() -> Result<(), CpuError> {
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
    fn jump_operation() -> Result<(), CpuError> {
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
    fn call_and_ret_operations() -> Result<(), CpuError> {
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
    fn ld_operation() -> Result<(), CpuError> {
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
    fn ldi_operation() -> Result<(), CpuError> {
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
    fn drw_operation_height_1() -> Result<(), CpuError> {
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
    fn drw_operation_height_3() -> Result<(), CpuError> {
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
    fn drw_operation_edge_clipping() -> Result<(), CpuError> {
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
    fn drw_operation_wrapping() -> Result<(), CpuError> {
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
