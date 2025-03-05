use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

pub type Buffer = Option<BufReader<File>>;

pub struct Rom {
    pub instructions: Vec<u8>,
    pub buffer: Buffer,
}

impl Default for Rom {
    fn default() -> Self {
        Self {
            instructions: Vec::new(),
            buffer: None,
        }
    }
}

// TODO fix api
// should call one func to get readable instructions back, everything else internal
// rename get_instructions to stor_instructions or something
// yeah this api is wonky, need to probably make this entirely a static suite of utility
// funcs
impl Rom {
    pub fn open_file(&mut self, file: &Path) {
        self.buffer = Some(BufReader::new(File::open(file).unwrap()));
    }

    pub fn get_instructions(&mut self) -> Result<(), &'static str> {
        if let Some(ref mut buffer) = self.buffer {
            for byte_or_error in buffer.bytes() {
                let byte = byte_or_error.map_err(|_| "Error reading byte")?;
                self.instructions.push(byte);
            }
            Ok(())
        } else {
            Err("No buffer found")
        }
    }

    pub fn print_instructions(&self) {
        for instruction in &self.instructions {
            println!("{:b}", instruction);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO Fix this test
    #[test]
    fn open() -> Result<(), &'static str> {
        let mut rom = Rom::default();
        rom.open_file(Path::new("./roms/test/IBM Logo.ch8"));
        rom.get_instructions()?;
        rom.print_instructions();
        assert_eq!(1, 1);
        Ok(())
    }
}
