"""Untrusted generated Python runner: unprivileged UID, seccomp, resource limits.

Fail closed if any isolation step is unavailable. No files/network/process creation.
"""
import ast
import builtins
import ctypes
import errno
import importlib
import json
import os
import resource
import signal
import sys

ALLOWED={'math','re','itertools','collections','functools','heapq','bisect','statistics',
         'operator','string','fractions','decimal','random','typing','cmath','copy','sys'}
FORBIDDEN={'open','exec','eval','compile','input','__import__','breakpoint','globals','locals',
           'vars','getattr','setattr','delattr','help','exit','quit'}

def validate(code):
    tree=ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n,ast.Import) and any(a.name not in ALLOWED for a in n.names):raise ValueError('unsupported import')
        if isinstance(n,ast.ImportFrom) and (n.module not in ALLOWED or n.level):raise ValueError('unsupported import')
        if isinstance(n,ast.Attribute) and n.attr.startswith('__'):raise ValueError('dunder attribute')
        if isinstance(n,ast.Name) and (n.id in FORBIDDEN or n.id.startswith('__')):raise ValueError('unsupported builtin')
    return compile(tree,'<candidate>','exec')

def sandbox():
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_load.argtypes=[ctypes.c_void_p]
    lib.seccomp_release.argtypes=[ctypes.c_void_p]
    libc=ctypes.CDLL(None,use_errno=True)
    os.setgroups([]);os.setgid(65534);os.setuid(65534)
    assert os.geteuid()==65534
    if libc.prctl(38,1,0,0,0)!=0:raise RuntimeError('no_new_privs failed')
    ctx=lib.seccomp_init(0x00050000|errno.EPERM)
    if not ctx:raise RuntimeError('seccomp init failed')
    calls=['read','write','close','fstat','newfstatat','lseek','brk','mmap','mprotect','munmap',
           'mremap','madvise','futex','clock_gettime','clock_nanosleep','nanosleep','gettimeofday',
           'exit','exit_group','rt_sigaction','rt_sigprocmask','rt_sigreturn','sigaltstack',
           'getpid','gettid','getuid','geteuid','getgid','getegid','getrandom','sched_yield',
           'getrusage','prlimit64','ioctl']
    for name in calls:
        nr=lib.seccomp_syscall_resolve_name(name.encode())
        if nr>=0 and lib.seccomp_rule_add(ctx,0x7fff0000,nr,0)!=0:raise RuntimeError('seccomp rule failed')
    if lib.seccomp_load(ctx)!=0:raise RuntimeError('seccomp load failed')
    lib.seccomp_release(ctx)

def main():
    payload=json.load(sys.stdin)
    modules={name:importlib.import_module(name) for name in ALLOWED}
    try:
        compiled=validate(payload['code'])
        setup=validate(payload.get('setup',''))
        tests=[validate(t) for t in payload['tests']]
    except (ValueError,SyntaxError) as e:
        print(json.dumps({'passed':False,'reason':type(e).__name__,'detail':str(e)[:200]}));return
    def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name not in modules or level:raise ImportError('unsupported import')
        return modules[name]
    safe={k:v for k,v in vars(builtins).items() if k not in FORBIDDEN and not k.startswith('__')}
    safe['__import__']=safe_import;safe['__build_class__']=builtins.__build_class__
    resource.setrlimit(resource.RLIMIT_CPU,(2,2))
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    signal.alarm(4)
    # Silence candidate output without losing the final result channel.
    result_fd=os.dup(1)
    null=os.open(os.devnull,os.O_WRONLY);os.dup2(null,1);os.dup2(null,2);os.close(null)
    try:sandbox()
    except BaseException:
        os.write(result_fd,b'{"passed":false,"reason":"sandbox_unavailable"}\n');return
    namespace={'__builtins__':safe,'__name__':'candidate'}
    try:
        exec(setup,namespace);exec(compiled,namespace)
        for test in tests:exec(test,namespace)
        result={'passed':True,'reason':'all_tests_passed'}
    except BaseException as e:result={'passed':False,'reason':type(e).__name__,'detail':str(e)[:200]}
    os.write(result_fd,(json.dumps(result)+'\n').encode())

if __name__=='__main__':main()
