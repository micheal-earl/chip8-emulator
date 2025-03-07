use crate::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Represents a loaded CHIP-8 ROM as a vector of opcodes
pub struct Rom {
    pub opcodes: Vec<u8>,
}

impl Rom {
    /// Loads a ROM from the given file path and returns a new Rom instance
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path.as_ref()).map_err(|e| {
            Error::Cpu(format!(
                "Failed to open ROM file {:?}: {}",
                path.as_ref(),
                e
            ))
        })?;
        let mut buf_reader = BufReader::new(file);
        let mut opcodes = Vec::new();
        buf_reader
            .read_to_end(&mut opcodes)
            .map_err(|e| Error::Cpu(format!("Error reading ROM file: {}", e)))?;
        Ok(Rom { opcodes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use std::path::Path;

    #[test]
    fn load_rom() -> Result<(), Error> {
        // Adjust the path as needed for testing
        let rom = Rom::from_path(Path::new("./roms/test/IBM Logo.ch8"))?;
        assert!(!rom.opcodes.is_empty());
        Ok(())
    }
}
