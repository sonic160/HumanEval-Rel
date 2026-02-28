import faulthandler
from typing import Optional
import traceback
import multiprocessing


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class SandboxCodeRunner:
    def __init__(self, timeout_seconds=10, timeout_warnings=False) -> None:
        # self.print_warning()
        self.TIMEOUT_SECONDS = timeout_seconds
        self.__timeout_warnings = timeout_warnings
        # self.reliability_guard() # TODO: Uncomment this line to enable the reliability guard

    def print_warning(self) -> None:
        warning_msg = "WARNING: This is NOT a security sandbox. Untrusted code, including, model-generated code, should not be blindly executed outside of one."
        print(bcolors.BOLD + bcolors.WARNING + warning_msg + bcolors.ENDC)

    def reliability_guard(self, maximum_memory_bytes: Optional[int] = None):
        """
        This disables various destructive functions and prevents the generated code
        from interfering with the test (e.g. fork bomb, killing other processes,
        removing filesystem files, etc.)

        WARNING
        This function is NOT a security sandbox. Untrusted code, including, model-
        generated code, should not be blindly executed outside of one. See the
        Codex paper for more information about OpenAI's code sandbox, and proceed
        with caution.

        Code taken fron OpenAI: https://github.com/openai/human-eval/blob/master/human_eval/execution.py
        """

        if maximum_memory_bytes is not None:
            import resource

            resource.setrlimit(
                resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
            )
            resource.setrlimit(
                resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
            )

        faulthandler.disable()

        import builtins

        builtins.exit = None
        builtins.quit = None

        import os

        # os.environ['OMP_NUM_THREADS'] = '1'

        os.kill = None
        os.system = None
        # os.putenv = None
        os.remove = None
        os.removedirs = None
        os.rmdir = None
        os.fchdir = None
        os.setuid = None
        os.fork = None
        os.forkpty = None
        os.killpg = None
        os.rename = None
        os.renames = None
        os.truncate = None
        os.replace = None
        os.unlink = None
        os.fchmod = None
        os.fchown = None
        os.chmod = None
        os.chown = None
        os.chroot = None
        os.fchdir = None
        os.lchflags = None
        os.lchmod = None
        os.lchown = None
        os.getcwd = None
        os.chdir = None

        import shutil

        shutil.rmtree = None
        shutil.move = None
        shutil.chown = None

        import subprocess

        subprocess.Popen = None  # type: ignore

        import sys

        sys.modules["ipdb"] = None
        sys.modules["joblib"] = None
        sys.modules["resource"] = None
        sys.modules["psutil"] = None
        sys.modules["tkinter"] = None

    def convert_to_bool(self, value: str | bool) -> bool:
        if isinstance(value, bool):
            return value
        if value == "pass":
            return True
        elif value == "fail":
            return False
        return True  # To allow running HumanEval challenges

    def __execute(self, code: str) -> tuple[bool, str | None]:
        """Execute code and return (passed, error_message)."""
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_exec_in_process, args=(code, result_queue), daemon=False
        )
        process.start()
        process.join(timeout=self.TIMEOUT_SECONDS)

        if process.is_alive():
            # Timeout - terminate the process
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            if self.__timeout_warnings:
                print(
                    bcolors.WARNING
                    + "\nWARNING[SandboxCodeRunner]: TIMEOUT error"
                    + bcolors.ENDC
                )
            return False, "Timeout: execution exceeded time limit"

        # Get result from queue
        try:
            test_result, error = result_queue.get_nowait()
            return self.convert_to_bool(test_result), error
        except Exception:
            return False, "Execution error: failed to retrieve result"

    def execute(self, code: str) -> tuple[bool, str | None]:
        """Public wrapper for executing arbitrary code. Returns (passed, error)."""
        return self.__execute(code)

    def run_tests(
        self, completion: str, tests: str, entry_point: str
    ) -> tuple[bool, str | None]:
        code = (
            "from typing import List, Callable\n"
            + "import numpy as np\n"
            + str(completion)
            + "\n"
            + tests
            + "\ntest = check("
            + entry_point
            + ")\n"
        )

        passed, error = self.__execute(code)
        return passed, error


def _exec_in_process(code: str, result_queue: multiprocessing.Queue) -> None:
    """Execute code in a separate process and put result in queue."""
    try:
        result = exec_with_context(code)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(("fail", f"{type(e).__name__}: {e}"))


def exec_with_context(code: str) -> tuple[str | bool, str | None]:
    """Execute code and return (result, error_message).

    result: 'pass', 'fail', or bool
    error_message: None if passed, error description if failed
    """
    context = {}
    try:
        exec(code, context)
        test_result = context["test"]
        if test_result == "pass" or test_result is True:
            return test_result, None
        else:
            return test_result, "Test assertion failed (returned 'fail')"
    except AssertionError as e:
        return "fail", f"AssertionError: {e}" if str(e) else "AssertionError"
    except Exception as e:
        tb = traceback.format_exc()
        return "fail", f"{type(e).__name__}: {e}\n{tb}"


if __name__ == "__main__":
    test = None
    runner = SandboxCodeRunner()
    func = "def p_f_interval(cdf: callable, t: float, delta_t: float) -> float:\n \treturn cdf(t + delta_t) - cdf(t)\n"
    tests = "def check(candidate):\n\timport numpy as np\n\tdef weibull_cdf(x, scale=2, shape=3):\n\t\treturn 1 - np.exp(-((x / scale) ** shape))\n\ttry:\n\t\tassert candidate(lambda x: weibull_cdf(x), 1, .5) == weibull_cdf(1.5) - weibull_cdf(1)\n\t\treturn 'pass'\n\texcept:\n\t\treturn 'fail'\n"
    pr = "test = check(p_f_interval)\n"

    print(runner.run_tests(pr, func, tests, "p_f_interval"))
