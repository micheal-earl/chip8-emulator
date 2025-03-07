use std::error::Error as StdError;
use std::fmt;
use std::io;
use std::sync;

// Import the minifb error type and rename it to avoid name conflicts
use minifb::Error as MinifbError;

/// A custom error type for the CHIP‑8 emulator
#[derive(Debug)]
pub enum Error {
    /// I/O errors (e.g. reading a ROM file)
    Io(io::Error),
    /// CPU errors – for example, invalid opcodes or memory access issues
    Cpu(String),
    /// Errors resulting from poisoned mutex locks
    Poison(String),
    /// Errors from the minifb graphics library
    Minifb(MinifbError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO Error: {}", e),
            Error::Cpu(msg) => write!(f, "CPU Error: {}", msg),
            Error::Poison(msg) => write!(f, "Mutex Poison Error: {}", msg),
            Error::Minifb(e) => write!(f, "Minifb Error: {}", e),
        }
    }
}

impl StdError for Error {}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}

impl<T> From<sync::PoisonError<T>> for Error {
    fn from(err: sync::PoisonError<T>) -> Self {
        Error::Poison(format!("Mutex poisoned: {}", err))
    }
}

impl From<MinifbError> for Error {
    fn from(err: MinifbError) -> Self {
        Error::Minifb(err)
    }
}

impl From<String> for Error {
    fn from(err: String) -> Self {
        Error::Cpu(err)
    }
}
