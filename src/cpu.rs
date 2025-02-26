pub struct Cpu {
    registers: [u8; 16],
    memory: [u8; 512],
    stack: [u16; 16],
    stack_pointer: usize,
    program_counter: usize,
    //display: [[u8; 64]; 32]
}

impl Default for Cpu {
    fn default() -> Self {
        Self {
            registers: [0; 16],
            memory: [0; 512],
            stack: [0; 16],
            stack_pointer: 0,
            program_counter: 0,
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_operation() {
        let mut cpu = Cpu::default();

        let reg = &mut cpu.registers;
        reg[0] = 5;
        reg[1] = 10;
        reg[2] = 10;
        reg[3] = 10;

        let mem = &mut cpu.memory;
        mem[0] = 0x80;
        mem[1] = 0x14;

        mem[2] = 0x80;
        mem[3] = 0x24;

        mem[4] = 0x80;
        mem[5] = 0x34;

        cpu.run();

        assert_eq!(cpu.registers[0], 35);
    }

    #[test]
    fn call_and_ret_operations() {
        let mut cpu = Cpu::default();

        let reg = &mut cpu.registers;
        reg[0] = 5;
        reg[1] = 10;

        let mem = &mut cpu.memory;
        mem[0x000] = 0x21;
        mem[0x001] = 0x00;

        mem[0x002] = 0x21;
        mem[0x003] = 0x00;

        mem[0x004] = 0x00;
        mem[0x005] = 0x00;

        //

        mem[0x100] = 0x80;
        mem[0x101] = 0x14;

        mem[0x102] = 0x80;
        mem[0x103] = 0x14;

        mem[0x104] = 0x00;
        mem[0x105] = 0xEE;

        cpu.run();

        assert_eq!(cpu.registers[0], 45);
    }

    #[test]
    fn mult_add_expression() {
        let mut cpu = Cpu::default();

        let reg = &mut cpu.registers;
        reg[0] = 5;
        reg[1] = 10;

        let mem = &mut cpu.memory;
        mem[0x000] = 0x21; // Call function as 0x100
        mem[0x001] = 0x00;

        mem[0x002] = 0x21; // Call function as 0x100
        mem[0x003] = 0x00;

        mem[0x004] = 0x00; // HALT
        mem[0x005] = 0x00;

        // Function
        mem[0x100] = 0x80; // ADD reg[1] to reg[0]
        mem[0x101] = 0x14;

        mem[0x102] = 0x80; // ADD reg[1] to reg[0]
        mem[0x103] = 0x14;

        mem[0x104] = 0x00; // Return
        mem[0x105] = 0xEE;

        cpu.run();

        assert_eq!(cpu.registers[0], 45);
    }
}
