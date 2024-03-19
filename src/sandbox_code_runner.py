class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
import sys
import faulthandler
from typing import Optional, Callable, Dict

class SandboxCodeRunner:
    def __init__(self):
        self.print_warning()
        #self.reliability_guard()

    def print_warning(self) -> None:
        warning_msg = 'WARNING: This is NOT a security sandbox. Untrusted code, including, model-generated code, should not be blindly executed outside of one.'
        print(bcolors.BOLD+bcolors.WARNING+ warning_msg + bcolors.ENDC)

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
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        

        faulthandler.disable()

        import builtins
        builtins.exit = None
        builtins.quit = None

        import os
        #os.environ['OMP_NUM_THREADS'] = '1'

        os.kill = None
        os.system = None
        #os.putenv = None
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
        sys.modules['ipdb'] = None
        sys.modules['joblib'] = None
        sys.modules['resource'] = None
        sys.modules['psutil'] = None
        sys.modules['tkinter'] = None 



    def convert_to_bool(self, value: str) -> bool:
        if value == 'pass':
            return True
        else:
            return False
        
    def run(self, code):
        # This method is used to run the code in a sandbox environment
        eval(code)
    def load_func(self, code: str):
        # This method is used to load the function from a string
        context = {}
        exec(code, context)
        
        return context['test']
        

if __name__ == '__main__':
    test = None
    runner = SandboxCodeRunner()
    func = 'def p_f_interval(cdf: callable, t: float, delta_t: float) -> float:\n \treturn cdf(t + delta_t) - cdf(t)\n'
    tests = 'def check(candidate):\n\timport numpy as np\n\tdef weibull_cdf(x, scale=2, shape=3):\n\t\treturn 1 - np.exp(-((x / scale) ** shape))\n\ttry:\n\t\tassert candidate(lambda x: weibull_cdf(x), 1, .5) == weibull_cdf(1.5) - weibull_cdf(1)\n\t\treturn \'pass\'\n\texcept:\n\t\treturn \'fail\'\n'
    pr = 'test = check(p_f_interval)\n'

    bl = runner.load_func(func+tests+pr)
    print(runner.convert_to_bool(bl))
    