pub mod cpu;
use cpu::Cpu;

fn main() -> Result<(), &'static str> {
    let mut cpu = Cpu::default();

    cpu.write_register(0x0, 5)?;
    cpu.write_register(0x1, 10)?;

    let memory_writes = [
        (0x000, 0x21), // Call function at 0x100
        (0x001, 0x00),
        (0x002, 0x21), // Call function at 0x100
        (0x003, 0x00),
        (0x004, 0x00), // HALT
        (0x005, 0x00),
        // Function
        (0x100, 0x80), // ADD reg[1] to reg[0]
        (0x101, 0x14),
        (0x102, 0x80), // ADD reg[1] to reg[0]
        (0x103, 0x14),
        (0x104, 0x00), // Return
        (0x105, 0xEE),
    ];

    cpu.write_memory_batch(&memory_writes)?;

    cpu.run();

    assert_eq!(cpu.read_register(0).unwrap(), 45u8);

    println!(
        "5 + (10 * 2) + (10 * 2) = {}",
        cpu.read_register(0).unwrap()
    );

    Ok(())
}
