import os
import shutil
from pathlib import Path
def clean_project_cache(root='.', dry_run=False):
    root = Path(root)
    cache_dirs = {'__pycache__', '.next', 'build', 'dist'}
    file_extensions = {'.pyc', '.pyo'}
    egg_info_suffix = '.egg-info'
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d != 'node_modules']
        for dirname in dirnames[:]:
            dir_full = dirpath / dirname
            if dirname in cache_dirs or dirname.endswith(egg_info_suffix):
                if dry_run:
                    print(f"[Dry Run] Would remove folder: {dir_full}")
                else:
                    shutil.rmtree(dir_full, ignore_errors=True)
                    print(f"Removed folder: {dir_full}")
                dirnames.remove(dirname)
        for filename in filenames:
            file_full = dirpath / filename
            if file_full.suffix in file_extensions:
                if dry_run:
                    print(f"[Dry Run] Would remove file: {file_full}")
                else:
                    try:
                        file_full.unlink()
                        print(f"Removed file: {file_full}")
                    except Exception as e:
                        print(f"Failed to remove {file_full}: {e}")
if __name__ == "__main__":
    clean_project_cache()