import sys
import faulthandler
import time
import os
from typing import Optional, Callable, Dict, List
import traceback
import multiprocessing


TIMEOUT_SECONDS = 10

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
    def __init__(self, timeout_warnings = False) -> None:
        self.print_warning()
        self.__timeout_warnings = timeout_warnings
        #self.reliability_guard() # TODO: Uncomment this line to enable the reliability guard

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

    def convert_to_bool(self, value: str) -> bool:
        if value == "pass":
            return True
        elif value == "fail":
            return False
        return True  # To allow running HumanEval challenges
    
    
    def __execute(self, *code_and_context) -> bool:
        try:
            with multiprocessing.Pool(processes=1) as pool:
                result = pool.apply_async(exec_with_context, [*code_and_context])
                
                try:
                    test_result = result.get(timeout=TIMEOUT_SECONDS)
                    return self.convert_to_bool(test_result)
                
                except multiprocessing.TimeoutError:
                    if self.__timeout_warnings:
                        print(bcolors.WARNING + "\nWARNING[SandboxCodeRunner]: TIMEOUT error (you may want to check the LLM's code)" + bcolors.ENDC)
                        print("\ncode that time out at execution:\n\n", *code_and_context, "\n\n\n")
                    return False
                
        except Exception as e:
            return False
        
    
    
    def run_tests(
        self, prompt: str, completion: str, tests: str, entry_point: str
    ) -> bool:
        #TODO: Delete the prompt argument
        # This method is used to run the tests in a sandbox environment
        code = (
            "from typing import List\n"
            + completion
            + "\n"
            + tests
            + "\ntest = check("
            + entry_point
            + ")\n"
        )

        try:
            test_result = self.__execute(code)
            return test_result, completion
        except ImportError as e:
            return False, completion+"\nImportError: "+str(e) 
        except Exception as e:
            return False, completion+"\nthe LLM's program failed to execute\n"+str(e)
def exec_with_context(code: str) -> bool:
        context = {}
        exec(code, context)
        return context['test']

if __name__ == "__main__":
    test = None
    runner = SandboxCodeRunner()
    func = "def p_f_interval(cdf: callable, t: float, delta_t: float) -> float:\n \treturn cdf(t + delta_t) - cdf(t)\n"
    tests = "def check(candidate):\n\timport numpy as np\n\tdef weibull_cdf(x, scale=2, shape=3):\n\t\treturn 1 - np.exp(-((x / scale) ** shape))\n\ttry:\n\t\tassert candidate(lambda x: weibull_cdf(x), 1, .5) == weibull_cdf(1.5) - weibull_cdf(1)\n\t\treturn 'pass'\n\texcept:\n\t\treturn 'fail'\n"
    pr = "test = check(p_f_interval)\n"

    print(runner.run_tests(pr, func, tests, "p_f_interval"))
