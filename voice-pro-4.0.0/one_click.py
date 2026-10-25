import os
import platform
import subprocess
import sys



class OneClick():
    """Bootstrap helper for the uv-managed environment.

    The heavy lifting (downloading uv, installing Python 3.12 and all
    dependencies from uv.lock) is done by start.bat / start.sh before this
    module runs.  This class only verifies the environment, offers a repair
    path (re-running `uv sync`), and launches the app.
    """

    script_dir = os.getcwd()

    install_dir = os.environ.get('INSTALL_DIR', os.path.join(script_dir, "installer_files"))
    env_path = os.environ.get('UV_PROJECT_ENVIRONMENT', os.path.join(script_dir, "installer_files", "env"))
    app_model_path = os.path.join(script_dir, "model")

    print("Info: Start 1-click ...")


    @classmethod
    def is_linux(cls):
        return sys.platform.startswith("linux")

    @classmethod
    def is_windows(cls):
        return sys.platform.startswith("win")

    @classmethod
    def is_macos(cls):
        return sys.platform.startswith("darwin")

    @classmethod
    def is_x86_64(cls):
        return platform.machine() == "x86_64"


    @classmethod
    def uv_exe(cls):
        uv = os.environ.get('UV_EXE')
        if uv and os.path.exists(uv):
            return uv
        name = "uv.exe" if cls.is_windows() else "uv"
        return os.path.join(cls.install_dir, "uv", name)


    @classmethod
    def env_python(cls):
        if cls.is_windows():
            return os.path.join(cls.env_path, "Scripts", "python.exe")
        return os.path.join(cls.env_path, "bin", "python")


    @classmethod
    def gpu_choice(cls):
        # GPU_CHOICE env var > choice saved by start/update scripts > CPU
        choice = os.environ.get("GPU_CHOICE", "").upper()
        if not choice:
            saved = os.path.join(cls.install_dir, "gpu_choice.txt")
            if os.path.exists(saved):
                choice = open(saved).read().strip().upper()
        return choice if choice in ("G", "C") else "C"


    @classmethod
    def torch_version(cls):
        try:
            from torch import __version__ as torver
            return torver
        except ImportError:
            return None


    @classmethod
    def oc_is_installed(cls):
        # Check that the key packages import from the current interpreter
        try:
            import json5   # noqa: F401
            import gradio  # noqa: F401
            import torch
            assert hasattr(torch, '_C') or hasattr(torch, '__version__')
            return True
        except Exception:
            return False


    @classmethod
    def oc_check_env(cls):
        # We must be running from the project-local uv environment
        env_prefix = os.path.abspath(cls.env_path)
        if os.path.abspath(sys.prefix) != env_prefix:
            print(f"Warning: not running from the expected environment.")
            print(f"  expected: {env_prefix}")
            print(f"  actual:   {sys.prefix}")

        # Ensure PYTHONPATH doesn't interfere with installed packages
        # Clear any torch-related paths that might cause conflicts
        pythonpath = os.environ.get('PYTHONPATH', '')
        if pythonpath:
            paths = pythonpath.split(os.pathsep)
            filtered_paths = [p for p in paths if not os.path.exists(os.path.join(p, 'torch', '_C'))]
            if len(filtered_paths) != len(paths):
                print("Warning: Removed PyTorch source paths from PYTHONPATH to avoid conflicts")
                os.environ['PYTHONPATH'] = os.pathsep.join(filtered_paths) if filtered_paths else ''

        # Check if we're in a PyTorch source directory (can cause import issues)
        torch_source_dir = os.path.join(os.getcwd(), "torch")
        if os.path.exists(os.path.join(torch_source_dir, "_C")):
            print("=" * 70)
            print("WARNING: PyTorch source directory detected in current path!")
            print(f"Found: {torch_source_dir}")
            print("This can cause PyTorch import errors. Remove or rename it.")
            print("=" * 70)


    @classmethod
    def oc_print_big_message(cls, message):
        message = message.strip()
        lines = message.split('\n')
        print("\n\n*******************************************************************")
        for line in lines:
            print("*", line)

        print("*******************************************************************\n\n")


    @classmethod
    def oc_run_cmd(cls, cmd, assert_success=False, environment=False, capture_output=False, env=None):
        # environment=True: make `python` resolve to the uv environment's interpreter
        if environment:
            env = dict(env or os.environ)
            env_bin = os.path.dirname(cls.env_python())
            env["PATH"] = env_bin + os.pathsep + env.get("PATH", "")
            env["VIRTUAL_ENV"] = cls.env_path

        # Run shell commands
        try:
            result = subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)

            # Assert the command ran successfully
            if assert_success and result.returncode != 0:
                print(f"Command '{cmd}' failed with exit status code '{str(result.returncode)}'.\n\nExiting now.\nTry running the start/update script again.")
                sys.exit(1)

            return result.returncode == 0
        except Exception as e:
            print(f"Command: '{cmd}' failed with {e}")
            return False


    @classmethod
    def oc_install_webui(cls, app_name: str, is_update=False):
        # Repair/update path: re-sync the environment from the committed lockfile.
        # (start.bat / start.sh normally do this before Python even starts.)
        extra = "gpu" if cls.gpu_choice() == "G" else "cpu"
        uv = cls.uv_exe()

        if not os.path.exists(uv):
            print(f"Error: uv not found at {uv}. Run the start script first. Exiting...")
            sys.exit(1)

        cls.oc_print_big_message(f"Syncing environment from uv.lock (extra: {extra})")
        cls.oc_run_cmd(f'"{uv}" sync --frozen --extra {extra}', assert_success=True)


    @classmethod
    def launch_webui(cls, app_file):
        print("Start the program...")
        cls.oc_run_cmd(f'"{cls.env_python()}" {app_file}')
