/// Returns a vec of all the paths within the directory. Files in this directory are assumed to
/// be named with an index of the file, followed by the file extension.
pub fn seq_files_in_dir(dir: &std::path::Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    // The read_dir function doesn't return paths in any particular order. So collect them and sort
    // them alphabetically to be sure they're handled in the right order.
    let mut paths = std::fs::read_dir(dir)?
        .map(|entry| Ok(entry?.path()))
        .collect::<std::io::Result<Vec<std::path::PathBuf>>>()?;
    paths.sort();
    Ok(paths)
}
