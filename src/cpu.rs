use std::thread;
use std::time;

type OpCode = u16;
type Instruction = (u8, u8, u8, u8, u8, u16);
// TODO consider type Display = [u8; 256], etc

pub const WIDTH: usize = 64;
pub const HEIGHT: usize = 32;

// CHIP-8 fonts consist of 16 characters, each defined by 5 bytes.
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
    registers: [u8; 16],
    memory: [u8; 4096],
    stack: [u16; 16],
    display: [u8; 256], //[[u8; 64]; 32],
    index: u16,
    stack_pointer: usize,
    program_counter: usize,
}

// TODO: delete these or make V0 - VF into constants
//const V0: u8 = 0u8;

impl Default for Cpu {
    fn default() -> Self {
        let mut cpu = Self {
            registers: [0; 16],
            memory: [0; 4096],
            stack: [0; 16],
            display: [0; 256],
            index: 0,
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
            println!(
                "{}, {}, {}",
                self.display[0], self.display[5], self.display[10]
            )
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
            (0xA, _, _, _) => self.ldi(nnn),
            (0xD, _, _, _) => self.drw(x, y, d),
            _ => todo!("catch all"),
        }
    }

    pub fn run(&mut self) {
        // TODO: Make execution 700hz (double check this)
        // TODO: Add 60hz timer for sound and delay
        let interval = DURATION_700HZ_IN_MICROS;
        let mut next_time = time::Instant::now() + interval;
        loop {
            let s = self.step();
            if !s {
                break;
            }

            // TODO the sleep calculation uses next_time - time::Instant::now(),
            // which might panic if the result is negative (if the CPU is busy)
            // maybe clamp the duration to zero.

            thread::sleep(next_time - time::Instant::now());
            next_time += interval;
        }
    }

    pub fn step(&mut self) -> bool {
        if self.program_counter >= 4095 {
            return false; // or return false if you want to exit
        }

        let opcode = self.fetch();
        let instruction = Self::decode(opcode);
        self.execute(instruction);
        true
    }

    // TODO use API for manipulating cpu object even for private functions
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

    /// (6xkk) LDI sets the value `nnn` into the index register (I register)
    fn ldi(&mut self, nnn: u16) {
        self.index = nnn;
    }

    /// (7xkk) ADD adds the value `kk` to register `Vx` and stores the sum in 'Vx'
    fn add(&mut self, x: u8, kk: u8) {
        self.registers[x as usize] += kk;
    }

    /// (2nnn) CALL sub-routine at `addr`
    fn call(&mut self, addr: u16) {
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

    /// Helper to get the value of an individual pixel from the bit-packed display.
    /// The display is stored as 256 u8’s, each holding 8 pixels.
    /// `pixel` is the overall pixel index (0..2047).
    fn get_display_pixel(&self, pixel: u16) -> u8 {
        let byte_index = (pixel / 8) as usize;
        let bit_index = 7 - (pixel % 8);
        (self.display[byte_index] >> bit_index) & 1
    }

    /// Display d-byte sprite starting at memory location I at (Vx, Vy).
    /// Sprites are XOR’d onto the display.
    /// If the sprite is partially off-screen, it is clipped (pixels outside are not drawn).
    /// If the sprite is completely off-screen, it wraps around (the starting coordinate is modulo adjusted).
    /// VF is set to 1 if any drawn pixel is erased.
    fn drw(&mut self, x: u8, y: u8, d: u8) {
        // Get the original coordinates from registers.
        let orig_x = self.read_register(x).unwrap() as usize;
        let orig_y = self.read_register(y).unwrap() as usize;
        let height = d as usize;

        // Determine if the entire sprite is off-screen:
        // We consider it "entirely off-screen" if the starting coordinate is not in bounds.
        let wrap = (orig_x >= WIDTH) || (orig_y >= HEIGHT);

        // If wrapping, adjust coordinates by modulo; otherwise, keep them for clipping.
        let x_coord = if wrap { orig_x % WIDTH } else { orig_x };
        let y_coord = if wrap { orig_y % HEIGHT } else { orig_y };

        // Reset collision flag.
        self.registers[0xF] = 0;
        let sprite_start = self.index as usize;

        // For each row of the sprite.
        for row in 0..height {
            // In clipping mode, skip rows that fall outside.
            if !wrap && (y_coord + row >= HEIGHT) {
                continue;
            }
            let sprite_byte = self.memory[sprite_start + row];

            // Each sprite row is 8 pixels wide.
            for col in 0..8 {
                // In clipping mode, skip columns that fall outside.
                if !wrap && (x_coord + col >= WIDTH) {
                    continue;
                }

                let sprite_pixel = (sprite_byte >> (7 - col)) & 1;
                if sprite_pixel == 1 {
                    // For wrapping mode, calculate coordinates modulo WIDTH/HEIGHT.
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
                    self.write_display(pixel_index, new_pixel == 1);
                }
            }
        }
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
            (0x0300, 0x8014), // 8014 ADD V1 to V0
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
            (0x0200, 0x2300), // 2nnn CALL subroutine at addr 0x300
            (0x0202, 0x2300), // 2nnn CALL subroutine at addr 0x300
            (0x0204, 0x1FFF), // 1nnn JUMP to nnn, in this case 0xFFF is the end of memory
            // Function
            (0x0300, 0x8014), // 8014 ADD reg[1] to reg[0]
            (0x0302, 0x8014), // 8014 ADD reg[1] to reg[0]
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
    fn ldi_operation() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        let instructions = [
            (0x0200, 0xA123), // Annn load I register with 0x123
            (0x0202, 0x1FFF), // 1nnn jump to 0xFFF
        ];

        cpu.write_instructions_batch(&instructions)?;

        cpu.run();

        assert_eq!(cpu.index, 0x123);

        Ok(())
    }

    // TODO write methods to read display and I
    // or maybe just use get_display_pixel to read display?
    #[test]
    fn drw_operation_height_1() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        // Program:
        // 00E0: CLS             (clear the screen)
        // 6000: LD V0, 0        (set V0 = 0; x-coordinate)
        // 6100: LD V1, 0        (set V1 = 0; y-coordinate)
        // A300: LDI 0x300       (set index register I = 0x300)
        // D011: DRW V0,V1,1     (draw 1-byte sprite at (V0,V1))
        // 1FFF: JP 0xFFF       (jump to exit)
        let instructions = [
            (0x0200, 0x00E0), // CLS
            (0x0202, 0x6000), // LD V0, 0
            (0x0204, 0x6100), // LD V1, 0
            (0x0206, 0xA300), // LDI 0x300
            (0x0208, 0xD011), // DRW V0,V1,1
            (0x020A, 0x1FFF), // JP 0xFFF (halt)
        ];
        cpu.write_instructions_batch(&instructions)?;
        // Write sprite data into memory at 0x300:
        // Sprite: 0x81 -> 0b10000001, so pixel at column 0 and 7 are on.
        cpu.write_memory(0x300, 0x81)?;

        cpu.run();

        // Verify that the sprite was drawn correctly.
        // Check pixel at (0,0):
        let pixel_index = 0u16; // row 0, col 0
        let byte_index = (pixel_index / 8) as usize;
        let bit_index = 7 - (pixel_index % 8);
        let pixel0 = (cpu.display[byte_index] >> bit_index) & 1;
        assert_eq!(pixel0, 1, "Pixel at (0,0) should be on");

        // Check pixel at (7,0):
        let pixel_index = 7u16; // row 0, col 7
        let byte_index = (pixel_index / 8) as usize;
        let bit_index = 7 - (pixel_index % 8);
        let pixel7 = (cpu.display[byte_index] >> bit_index) & 1;
        assert_eq!(pixel7, 1, "Pixel at (7,0) should be on");

        // Verify pixels in between (columns 1 through 6) are off.
        for col in 1..7 {
            let pixel_index = col as u16; // row 0, col = col
            let byte_index = (pixel_index / 8) as usize;
            let bit_index = 7 - (pixel_index % 8);
            let pixel = (cpu.display[byte_index] >> bit_index) & 1;
            assert_eq!(pixel, 0, "Pixel at ({},0) should be off", col);
        }

        // The collision flag (VF) should remain 0.
        assert_eq!(
            cpu.read_register(0xF).unwrap(),
            0,
            "VF should be 0 for height 1 sprite"
        );
        Ok(())
    }

    #[test]
    fn drw_operation_height_3() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        // Program:
        // 00E0: CLS              (clear the screen)
        // 600A: LD V0, 10       (set V0 = 10; x-coordinate)
        // 6105: LD V1, 5        (set V1 = 5; y-coordinate)
        // A300: LDI 0x300       (set I = 0x300)
        // D013: DRW V0,V1,3     (draw 3-byte sprite at (V0,V1))
        // 1FFF: JP 0xFFF       (halt)
        let instructions = [
            (0x0200, 0x00E0), // CLS
            (0x0202, 0x600A), // LD V0, 10
            (0x0204, 0x6105), // LD V1, 5
            (0x0206, 0xA300), // LDI 0x300
            (0x0208, 0xD013), // DRW V0,V1,3
            (0x020A, 0x1FFF), // JP 0xFFF (halt)
        ];
        cpu.write_instructions_batch(&instructions)?;
        // Write sprite data into memory at 0x300:
        // Row 0: 0x3C -> 0b00111100
        // Row 1: 0x42 -> 0b01000010
        // Row 2: 0x81 -> 0b10000001
        cpu.write_memory(0x300, 0x3C)?;
        cpu.write_memory(0x301, 0x42)?;
        cpu.write_memory(0x302, 0x81)?;

        cpu.run();

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
                let byte_index = (pixel_index / 8) as usize;
                let bit_index = 7 - (pixel_index % 8);
                let pixel_val = (cpu.display[byte_index] >> bit_index) & 1;
                assert_eq!(
                    pixel_val, expected[row][col],
                    "Mismatch at row {} col {}",
                    row, col
                );
            }
        }

        // Collision flag (VF) should be 0 since no pixels were erased
        assert_eq!(
            cpu.read_register(0xF).unwrap(),
            0,
            "VF should be 0 for height 3 sprite"
        );
        Ok(())
    }
    #[test]
    fn drw_operation_edge_clipping() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        // Program:
        // 00E0: CLS              (clear the screen)
        // 6000: LD V0, 60       (set V0 = 60; x-coordinate is within display but near right edge)
        // 6105: LD V1, 5        (set V1 = 5; y-coordinate is within display)
        // A300: LDI 0x300       (set index register I = 0x300)
        // D011: DRW V0,V1,1     (draw 1-byte sprite at (V0,V1))
        // 1FFF: JP 0xFFF       (halt)
        let instructions = [
            (0x0200, 0x00E0), // CLS
            (0x0202, 0x603C), // LD V0, 60    (0x6000 | 60 = 0x603C)
            (0x0204, 0x6105), // LD V1, 5     (0x6100 | 5  = 0x6105)
            (0x0206, 0xA300), // LDI 0x300
            (0x0208, 0xD011), // DRW V0,V1,1
            (0x020A, 0x1FFF), // JP 0xFFF (halt)
        ];
        cpu.write_instructions_batch(&instructions)?;

        // Write sprite data: one byte sprite: 0xFF (all 8 pixels on)
        // When drawn at (60,5) on a 64-wide display, only pixels at x=60..63 should be drawn.
        cpu.write_memory(0x300, 0xFF)?;

        cpu.run();

        // For row 5, check that only columns 60 to 63 are on, and all other columns are off.
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
            cpu.read_register(0xF).unwrap(),
            0,
            "VF should be 0 when clipping without collision"
        );
        Ok(())
    }

    #[test]
    fn drw_operation_wrapping() -> Result<(), &'static str> {
        let mut cpu = Cpu::default();

        // Program:
        // 00E0: CLS              (clear the screen)
        // 6000: LD V0, 70       (set V0 = 70; completely off-screen horizontally)
        // 610A: LD V1, 10       (set V1 = 10; y-coordinate is in range)  (0x6100 | 10 = 0x610A)
        // A300: LDI 0x300       (set index register I = 0x300)
        // D011: DRW V0,V1,1     (draw 1-byte sprite at (V0,V1) – should wrap horizontally)
        // 1FFF: JP 0xFFF       (halt)
        let instructions = [
            (0x0200, 0x00E0),      // CLS
            (0x0202, 0x6000 | 70), // LD V0, 70  (70 decimal)
            (0x0204, 0x6100 | 10), // LD V1, 10 (10 decimal)
            (0x0206, 0xA300),      // LDI 0x300
            (0x0208, 0xD011),      // DRW V0,V1,1
            (0x020A, 0x1FFF),      // JP 0xFFF (halt)
        ];
        cpu.write_instructions_batch(&instructions)?;
        // Write sprite data: one byte sprite: 0xFF (all 8 pixels on).
        cpu.write_memory(0x300, 0xFF)?;

        cpu.run();

        // Since V0=70, wrapping yields 70 % 64 = 6.
        // For row 10, we expect columns 6..13 to be drawn (with no clipping since 13 < 64).
        for x in 6..14 {
            let pixel_index = (10 * WIDTH + x) as u16;
            let pixel = cpu.get_display_pixel(pixel_index);
            assert_eq!(pixel, 1, "Pixel at ({},10) should be on due to wrapping", x);
        }
        // Also check that a pixel outside the drawn region remains off.
        let outside_index = (10 * WIDTH + 5) as u16;
        assert_eq!(
            cpu.get_display_pixel(outside_index),
            0,
            "Pixel at (5,10) should be off"
        );
        // Collision flag should be 0.
        assert_eq!(
            cpu.read_register(0xF).unwrap(),
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
