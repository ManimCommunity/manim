from subprocess import run


def capture(command, cwd=None, command_input=None):
    p = run(command, cwd=cwd, input=command_input, capture_output=True, text=True)
    out, err = p.stdout, p.stderr
    return out, err, p.returncode
