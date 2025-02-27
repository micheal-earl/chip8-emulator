pub mod cpu;
use cpu::Cpu;

fn main() -> Result<(), &'static str> {
    let mut cpu = Cpu::default();

    cpu.write_register(0x0, 5)?;
    cpu.write_register(0x1, 10)?;

    let memory_writes = [
        (0x0200, 0x23), // Call function at 0x100
        (0x0201, 0x00),
        (0x0202, 0x23), // Call function at 0x100
        (0x0203, 0x00),
        (0x0204, 0x00), // HALT
        (0x0205, 0x00),
        // Function
        (0x0300, 0x80), // ADD reg[1] to reg[0]
        (0x0301, 0x14),
        (0x0302, 0x80), // ADD reg[1] to reg[0]
        (0x0303, 0x14),
        (0x0304, 0x00), // Return
        (0x0305, 0xEE),
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
