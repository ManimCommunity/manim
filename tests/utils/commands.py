import subprocess


def capture(command, cwd=None, shell=False):
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf8",
        cwd=cwd,
        shell=shell,
    )
    out, err = proc.communicate()
    return out, err, proc.returncode
