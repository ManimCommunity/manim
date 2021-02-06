import subprocess


def capture(command, cwd=None, shell=False, command_input=None):
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        encoding="utf8",
        cwd=cwd,
        shell=shell,
    )
    out, err = proc.communicate(command_input)
    return out, err, proc.returncode
