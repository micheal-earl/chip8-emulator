pub struct Cpu {
    registers: [u8; 16],
    memory: [u8; 512],
    stack: [u16; 16],
    display: [[u8; 64]; 32],
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

impl Default for Cpu {
    fn default() -> Self {
        let mut cpu = Self {
            registers: [0; 16],
            memory: [0; 512],
            stack: [0; 16],
            display: [[0; 64]; 32],
            stack_pointer: 0,
            program_counter: 0,
        };

        cpu.write_fonts_to_memory();

        cpu
    }
}

impl Cpu {
    fn read_opcode(&self) -> u16 {
        let pc = self.program_counter;
        let op_high_byte = self.memory[pc] as u16;
        let op_low_byte = self.memory[pc + 1] as u16;

        op_high_byte << 8 | op_low_byte
    }

    pub fn run(&mut self) {
        loop {
            let opcode = self.read_opcode();
            self.program_counter += 2;

            // 0x73EE
            // 73 = High Byte, EE = Low Byte
            // 7 = High Nibble, 3 = Low Nibble, same for EE

            // Var, Bit-length, Location,                  Description,
            // n    4           low byte, low nibble       Number of bytes
            // x    4           high byte, low nibble      CPU register
            // y    4           low byte, high nibble      CPU register
            // c    4           high byte, high nibble     Opcode group
            // d    4           low byte, low nibble       Opcode subgroup
            // kk   8           low byte, both nibbles     Integer
            // nnn  12          high byte, low nibble      Memory address
            //                and low byte, both nibbles

            let c = ((opcode & 0xF000) >> 12) as u8;
            let x = ((opcode & 0x0F00) >> 8) as u8;
            let y = ((opcode & 0x00F0) >> 4) as u8;
            let d = (opcode & 0x000F) as u8; // op minor
            let _kk = (opcode & 0x00FF) as u8;
            let nnn = opcode & 0x0FFF; // addr

            match (c, x, y, d) {
                (0, 0, 0, 0) => {
                    return;
                }
                //(0, 0, 0xE, 0) => { /* Clear Screen op */ }
                (0, 0, 0xE, 0xE) => self.ret(),
                (0x1, _, _, _) => self.jmp(nnn),
                (0x2, _, _, _) => self.call(nnn),
                (0x2, _, _, _) => todo!(), //self.se(x, kk),
                (0x2, _, _, _) => todo!(), //self.sne(x, kk),
                (0x2, _, _, _) => todo!(), //self.se(x, y),
                (0x2, _, _, _) => todo!(), //self.ld(x, kk),
                (0x8, _, _, _) => match d {
                    4 => self.add_xy(x, y),
                    _ => todo!("opcode {:04x}", opcode),
                },
                _ => todo!("opcode {:04x}", opcode),
            }
        }
    }

    /// (1nnn) JUMP to `addr`
    fn jmp(&mut self, addr: u16) {
        todo!();
    }

    /// (2xkk) SE  **S**tore if **e**qual
    fn se(&mut self, xv: u8, xkk: u8) {
        todo!();
    }

    /// (2xkk) SNE  **S**tore if **n**ot **e**qual
    fn sne(&mut self, xv: u8, xkk: u8) {
        todo!();
    }

    /// (6xkk) LD sets the value `kk` into register `vx`
    fn ld(&mut self, xv: u8, xkk: u8) {
        todo!();
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

    fn write_fonts_to_memory(&mut self) {
        // The font data occupies 80 bytes starting at memory address 0x50.
        // We don't use write_memory_batch here as that would be slower
        let start = 0x50;
        let end = start + FONT_DATA.len();
        self.memory[start..end].copy_from_slice(&FONT_DATA);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_operation() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        // Write initial register values
        cpu.write_register(0, 5)?;
        cpu.write_register(1, 10)?;
        cpu.write_register(2, 10)?;
        cpu.write_register(3, 10)?;

        // Write opcodes into memory using write_memory_batch.
        let memory_writes = [
            (0x0000, 0x80), // Opcode 0x8014: ADD reg[1] to reg[0]
            (0x0001, 0x14),
            (0x0002, 0x80), // Opcode 0x8024: ADD reg[2] to reg[0]
            (0x0003, 0x24),
            (0x0004, 0x80), // Opcode 0x8034: ADD reg[3] to reg[0]
            (0x0005, 0x34),
        ];

        cpu.write_memory_batch(&memory_writes)?;

        cpu.run();

        assert_eq!(cpu.read_register(0).unwrap(), 35);

        Ok(())
    }

    #[test]
    fn call_and_ret_operations() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        // Write initial register values
        cpu.write_register(0x0, 5)?;
        cpu.write_register(0x1, 10)?;

        let memory_writes = [
            (0x000, 0x21), // Opcode 0x2100 CALL function at 0x100
            (0x001, 0x00),
            (0x002, 0x21), // Opcode 0x2100 CALL function at 0x100
            (0x003, 0x00),
            (0x004, 0x00), // HALT
            (0x005, 0x00),
            // Function
            (0x100, 0x80), // Opcode 0x8014: ADD reg[1] to reg[0]
            (0x101, 0x14),
            (0x102, 0x80), // Opcode 0x8014: ADD reg[1] to reg[0]
            (0x103, 0x14),
            (0x104, 0x00), // RETURN
            (0x105, 0xEE),
        ];

        cpu.write_memory_batch(&memory_writes)?;

        cpu.run();

        assert_eq!(cpu.read_register(0).unwrap(), 45);

        Ok(())
    }

    #[test]
    fn font_data_written_correctly() {
        let cpu = Cpu::default();
        let expected_font_data: [u8; 80] = FONT_DATA;

        let start = 0x50;
        let end = start + expected_font_data.len();
        assert_eq!(&cpu.memory[start..end], &expected_font_data);
    }
}
