use std::thread;
use std::time;

type OpCode = u16;
type Instruction = (u8, u8, u8, u8, u8, u16);
// TODO consider type Display = [u8; 256], etc

pub struct Cpu {
    registers: [u8; 16],
    memory: [u8; 4096],
    stack: [u16; 16],
    display: [u8; 256], //[[u8; 64]; 32],
    stack_pointer: usize,
    program_counter: usize,
}

// PICO-8 fonts consist of 16 characters, each defined by 5 bytes.
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

// TODO: delete these or make V0 - VF into constants
//const V0: u8 = 0u8;

impl Default for Cpu {
    fn default() -> Self {
        let mut cpu = Self {
            registers: [0; 16],
            memory: [0; 4096],
            stack: [0; 16],
            display: [0; 256],
            stack_pointer: 0,
            program_counter: 0x0200,
        };

        cpu.write_fonts_to_memory();

        cpu
    }
}

impl Cpu {
    fn fetch(&mut self) -> OpCode {
        let pc = self.program_counter;
        let op_high_byte = self.memory[pc] as u16;
        let op_low_byte = self.memory[pc + 1] as u16;

        // TODO: Remove debug
        let out = op_high_byte << 8 | op_low_byte;
        if out != 0 {
            println!("{}, {:#04x}", self.program_counter, &out);
        }
        // DEBUG ^

        self.program_counter += 2;

        op_high_byte << 8 | op_low_byte
    }

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

    fn execute(&mut self, decoded: Instruction) {
        let (c, x, y, d, kk, nnn) = decoded;
        match (c, x, y, d) {
            (0x0, 0x0, 0x0, 0x0) => return,     // 0000
            (0x0, 0x0, 0xE, 0x0) => self.cls(), // 00E0 Clear the screen
            (0x0, 0x0, 0xE, 0xE) => self.ret(), // 00EE Return from subroutine
            (0x1, _, _, _) => self.jmp(nnn),    // 1nnn Jump to location nnn
            (0x2, _, _, _) => self.call(nnn),   // Call subroutine at location nnn
            (0x3, _, _, _) => self.se(x, kk),   // 3xkk Skip next instruction if Vx == kk
            (0x4, _, _, _) => self.sne(x, kk),  // 4xkk Skip next instruvtion if Vx != kk
            (0x5, _, _, 0x0) => self.se(x, y),  // 5xy0 Skip next instruction if Vx == Vy
            (0x6, _, _, _) => self.ld(x, kk),   // 6xkk Write kk to Vx
            (0x7, _, _, _) => self.add(x, kk),  // 7xkk Add kk to Vx, write result to Vx
            (0x8, _, _, _) => match d {
                4 => self.add_xy(x, y),
                _ => todo!("d not 4"),
            },
            _ => todo!("catch all"),
        }
    }

    pub fn run(&mut self) {
        // TODO: Make execution 700hz (double check this)
        // TODO: Add 60hz timer for sound and delay
        let interval = time::Duration::from_micros(1);
        let mut next_time = time::Instant::now() + interval;
        loop {
            if self.program_counter >= 4095 {
                break;
            }

            let opcode = self.fetch();
            let instruction = Self::decode(opcode);
            self.execute(instruction);

            thread::sleep(next_time - time::Instant::now());
            next_time += interval;
        }
    }

    /// (00E0) CLEAR the display
    fn cls(&mut self) {
        self.display = [0; 256];
    }

    /// (1nnn) JUMP to `addr`
    fn jmp(&mut self, addr: u16) {
        self.program_counter = addr as usize;
    }

    /// (2xkk) SE  _S_kip if _e_qual
    fn se(&mut self, _vx: u8, _kk: u8) {
        todo!("se");
    }

    /// (2xkk) SNE  _S_kip if _n_ot _e_qual
    fn sne(&mut self, _vx: u8, _kk: u8) {
        todo!("sne");
    }

    /// (6xkk) LD sets the value `kk` into register `Vx`
    fn ld(&mut self, vx: u8, kk: u8) {
        self.registers[vx as usize] = kk;
    }

    /// (7xkk) ADD adds the value `kk` to register `Vx` and stores the sum in 'Vx'
    fn add(&mut self, x: u8, kk: u8) {
        self.registers[x as usize] += kk;
    }

    /// (2nnn) CALL sub-routine at `addr`
    fn call(&mut self, addr: u16) {
        if self.stack_pointer > self.stack.len() {
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
    fn add_xy(&mut self, x: u8, y: u8) {
        let arg1 = self.registers[x as usize];
        let arg2 = self.registers[y as usize];

        let (val, overflow) = arg1.overflowing_add(arg2);
        self.registers[x as usize] = val;

        if overflow {
            self.registers[0xF] = 1;
        } else {
            self.registers[0xF] = 0;
        }
    }

    pub fn write_display(&mut self, pixel: u16, value: bool) {
        // Calculate which byte holds the pixel.
        let byte_index = (pixel / 8) as usize;
        // Calculate the bit position within that byte.
        // We assume bit 7 is the leftmost pixel, so we subtract the remainder from 7.
        let bit_index = 7 - (pixel % 8);

        // Check if the byte index is valid for our display buffer.
        if byte_index >= self.display.len() {
            // TODO add error
            return;
        }

        if value {
            // Set the bit to 1 to turn the pixel on.
            self.display[byte_index] |= 1 << bit_index;
        } else {
            // Clear the bit to 0 to turn the pixel off.
            self.display[byte_index] &= !(1 << bit_index);
        }
    }

    pub fn read_display(&self) -> &[u8; 256] {
        &self.display
    }

    pub fn read_register(&self, address: u8) -> Option<u8> {
        self.registers.get(address as usize).copied()
    }

    pub fn write_register(&mut self, address: u8, val: u8) -> Result<(), &'static str> {
        let address_usize = address as usize;
        if address_usize < self.registers.len() {
            self.registers[address_usize] = val;
            Ok(())
        } else {
            Err("Register index out of bounds")
        }
    }

    pub fn read_memory(&self, address: u8) -> Option<u8> {
        self.memory.get(address as usize).copied()
    }

    pub fn write_memory(&mut self, address: u16, val: u8) -> Result<(), &'static str> {
        let index = address as usize;
        if index < self.memory.len() {
            self.memory[index] = val;
            Ok(())
        } else {
            Err("Memory address out of bounds")
        }
    }

    pub fn write_memory_batch(&mut self, writes: &[(u16, u8)]) -> Result<(), &'static str> {
        for &(address, value) in writes {
            self.write_memory(address, value)?;
        }
        Ok(())
    }

    pub fn write_instruction(
        &mut self,
        address: u16,
        instruction: u16,
    ) -> Result<(), &'static str> {
        let index = address as usize;
        if index + 1 >= self.memory.len() {
            return Err("Memory address out of bounds");
        }
        // Split the instruction into high and low bytes.
        self.memory[index] = (instruction >> 8) as u8;
        self.memory[index + 1] = (instruction & 0xFF) as u8;
        Ok(())
    }

    pub fn write_instructions_batch(
        &mut self,
        instructions: &[(u16, u16)],
    ) -> Result<(), &'static str> {
        for &(address, instruction) in instructions {
            self.write_instruction(address, instruction)?;
        }
        Ok(())
    }

    fn write_fonts_to_memory(&mut self) {
        // The font data occupies 80 bytes starting at memory address 0x50.
        // We don't use write_memory_batch here as that would be slower
        let start = 0x00;
        let end = start + FONT_DATA.len();
        self.memory[start..end].copy_from_slice(&FONT_DATA);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO comments on tests

    #[test]
    fn add_xy_operation() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        // Write initial register values
        cpu.write_register(0x0000, 5)?;
        cpu.write_register(0x0001, 10)?;
        cpu.write_register(0x0002, 15)?;
        cpu.write_register(0x0003, 7)?;

        // Write opcodes into memory using write_memory_batch.
        let instructions = [
            (0x0200, 0x8014), // 8014: ADD V1 to V0
            (0x0202, 0x8024), // 8024: ADD V2 to V0
            (0x0204, 0x8034), // 8034: ADD V3 to V0
        ];

        cpu.write_instructions_batch(&instructions)?;

        cpu.run();

        assert_eq!(cpu.read_register(0).unwrap(), 37);

        Ok(())
    }

    #[test]
    fn add_operation() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        // Set an initial value in register V1.
        cpu.write_register(1, 20)?;

        let instructions = [
            (0x0200, 0x7105), // 7xkk Add 0x05 to register V1 (20 + 5 = 25)
            (0x0202, 0x1FFF), // 1nnn Jump to 0xFFF
        ];

        cpu.write_instructions_batch(&instructions)?;

        cpu.run();

        // Verify that register V1 now holds the value 25.
        assert_eq!(cpu.read_register(1).unwrap(), 25);

        Ok(())
    }

    #[test]
    fn jump_operation() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        cpu.write_register(0x0000, 5)?;
        cpu.write_register(0x0001, 7)?;

        let instructions = [
            (0x0200, 0x1300), // 1nnn JUMP to nnn
            (0x0300, 0x8014), // 8014: ADD V1 to V0
        ];

        cpu.write_instructions_batch(&instructions)?;

        cpu.run();

        assert_eq!(cpu.read_register(0).unwrap(), 12);

        Ok(())
    }

    #[test]
    fn call_and_ret_operations() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        // Write initial register values
        cpu.write_register(0x0000, 5)?;
        cpu.write_register(0x0001, 10)?;

        let instructions = [
            (0x0200, 0x2300), // 2nnn CALL subroutine at addr 300
            (0x0202, 0x2300), // 2nnn CALL subroutine at addr 300
            (0x0204, 0x1FFF), // 1nnn JUMP to nnn, in this case FFF is the end of memory
            // Function
            (0x0300, 0x8014), // Opcode 0x8014: ADD reg[1] to reg[0]
            (0x0302, 0x8014), // Opcode 0x8014: ADD reg[1] to reg[0]
            (0x0304, 0x00EE), // RETURN
        ];

        cpu.write_instructions_batch(&instructions)?;

        cpu.run();

        assert_eq!(cpu.read_register(0).unwrap(), 45);

        Ok(())
    }

    #[test]
    fn ld_operation() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        let instructions = [
            (0x0200, 0x63AB), // 6xkk load V3 with 0xAB.
            (0x0202, 0x1FFF), // 1nnn jump to 0xFFF
        ];

        cpu.write_instructions_batch(&instructions)?;

        cpu.run();

        assert_eq!(cpu.read_register(3).unwrap(), 0xAB);

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
