use std::{fs::File, io::Write};

fn get_output_file(path: &str, silent: bool) -> (Option<std::fs::File>, u64) {
    let mut transferred: u64 = 0;
    let mut file = None;
    if path != "stdout" {
        if std::path::Path::new(path).exists() {
            if !silent {
                println!("File exists. Resuming.");
            }
            file = Some(std::fs::OpenOptions::new().append(true).open(path).unwrap());

            let file_size = std::fs::metadata(path).unwrap().len();
            transferred = file_size;
        } else {
            if !silent {
                println!("Writing to new file.");
            }
            file = Some(
                File::create(path)
                    .map_err(|_| format!("Failed to create file '{path}'"))
                    .unwrap(),
            );
        }
    }
    (file, transferred)
}

pub fn get_output(path: &str, silent: bool) -> (Box<dyn Write + Send>, u64) {
    let (file, transferred) = get_output_file(path, silent);
    let output: Box<dyn Write + Send> = Box::new(std::io::BufWriter::new(match path {
        "stdout" => Box::new(std::io::stdout()) as Box<dyn Write + Send>,
        _ => Box::new(file.unwrap()) as Box<dyn Write + Send>,
    }));

    (output, transferred)
}

#[test]
fn test_get_output_file_file_is_none_when_stdout() {
    let is_silet = true;
    let (file, _) = get_output_file("stdout", is_silet);
    assert!(file.is_none());
}

#[test]
fn test_get_output_file_pos_is_zero_when_stdout() {
    let is_silet = true;
    let (_, position) = get_output_file("stdout", is_silet);
    assert_eq!(position, 0);
}

#[test]
fn test_get_output_file_file_is_none_when_newfile() {
    let is_silet = true;
    let filename = "test_get_output_file_file_is_none_when_newfile";

    let (file, _) = get_output_file(filename, is_silet);

    assert!(file.is_some());
    std::fs::remove_file(filename).unwrap();
}

#[test]
fn test_get_output_file_file_is_none_when_newfile_and_not_silent() {
    let is_silet = false;
    let filename = "test_get_output_file_file_is_none_when_newfile_and_not_silent";

    let (file, _) = get_output_file(filename, is_silet);

    assert!(file.is_some());
    std::fs::remove_file(filename).unwrap();
}

#[test]
fn test_get_output_file_file_is_none_when_existingfile_and_not_silent() {
    use std::io::Write;
    let is_silet = false;
    let filename = "test_get_output_file_file_is_none_when_existingfile_and_not_silent";
    let expected_position_byte = 4;
    let mut file = File::create(filename).unwrap();
    file.write_all(b"1234").unwrap();

    let (_, position) = get_output_file(filename, is_silet);

    assert_eq!(position, expected_position_byte);
    std::fs::remove_file(filename).unwrap();
}
