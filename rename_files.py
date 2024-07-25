import os


def rename_files(directory, new_name_pattern):
    """
    Rename files in the specified directory based on the new name pattern.

    Parameters:
    - directory (str): The path to the directory containing the files to be renamed.
    - new_name_pattern (str): The new name pattern for the files, e.g., "file_{:03d}.txt".
    """
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Sort files (optional, if order matters)
    files.sort()

    # Rename each file
    for index, file_name in enumerate(files):
        # Construct the new file name
        new_name = new_name_pattern.format(index + 1)

        # Construct full file paths
        old_path = os.path.join(directory, file_name)
        new_path = os.path.join(directory, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {old_path} -> {new_path}')


# Example usage
directory = 'non_UAV_sounds/'
new_name_pattern = 'sound_{:04d}.wav'
rename_files(directory, new_name_pattern)
