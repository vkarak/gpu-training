import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest


class OpenACCBaseTest(RegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['dom:gpu', 'daint:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.sourcesdir = os.path.join(self.prefix, '../solutions')
        self.modules = ['craype-accel-nvidia60']
        self.maintainers = ['karakasis<at>cscs.ch']

    def compile(self):
        self.current_environ.propagate = False
        super().compile()


class AXPYExample(OpenACCBaseTest):
    def __init__(self, lang='c', **kwargs):
        super().__init__('axpy_%s_example' % lang, **kwargs)
        if lang == 'fortran':
            self.executable = './axpy/axpy.openacc.fort'
        else:
            self.executable = './axpy/axpy.openacc'

        self.sourcepath = 'axpy/'
        self.sanity_patterns = sn.assert_found('PASSED', self.stdout)


class DiffusionExample(OpenACCBaseTest):
    def __init__(self, version, **kwargs):
        super().__init__('diffusion_%s_example' %
                         version.replace('+', '_'), **kwargs)
        self.sourcepath = 'diffusion/'
        self.executable = ('./diffusion/diffusion2d.%s' %
                           version.replace('+', '.'))
        if 'mpi' in version:
            # PGI 17.X, 18.X do not like mixing CUDA and OpenACC!
            self.valid_prog_environs = ['PrgEnv-cray']
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.variables = {
                'CRAY_CUDA_MPS': '1',
                'MPICH_RDMA_ENABLED_CUDA': '1'
            }

        self.sanity_patterns = sn.assert_found('writing to output',
                                               self.stdout)


class GemmExample(OpenACCBaseTest):
    def __init__(self, **kwargs):
        super().__init__('gemm_example', **kwargs)
        self.sourcepath = 'gemm/'
        self.executable = './gemm/gemm.openacc'
        self.num_cpus_per_task = 12
        self.variables = {'OMP_NUM_THREADS': str(self.num_cpus_per_task)}
        self.sanity_patterns = sn.assert_eq(
            3, sn.count(sn.extractall('success', self.stdout))
        )


class BlurExample(OpenACCBaseTest):
    def __init__(self, version, **kwargs):
        super().__init__(
            'blur_%s_example' % version.replace('+', '_'), **kwargs)
        self.sourcepath = 'shared/'
        self.executable = './shared/blur.%s' % version.replace('+', '.')
        self.sanity_patterns = sn.assert_found('success', self.stdout)


class DotExample(OpenACCBaseTest):
    def __init__(self, version, **kwargs):
        super().__init__(
            'dot_%s_example' % version.replace('+', '_'), **kwargs)
        self.sourcepath = 'shared/'
        self.executable = './shared/dot.%s' % version.replace('+', '.')
        self.sanity_patterns = sn.assert_found('success', self.stdout)


def _get_checks(**kwargs):
    return [AXPYExample(**kwargs),
            AXPYExample('fortran', **kwargs),
            DiffusionExample('omp', **kwargs),
            BlurExample('openacc', **kwargs),
            BlurExample('openacc+fort', **kwargs),
            DiffusionExample('openacc', **kwargs),
            DiffusionExample('openacc+cuda', **kwargs),
            DiffusionExample('openacc+cuda+mpi', **kwargs),
            DiffusionExample('openacc+mpi', **kwargs),
            DiffusionExample('openacc+fort', **kwargs),
            DiffusionExample('openacc+fort+mpi', **kwargs),
            DotExample('openacc', **kwargs),
            DotExample('openacc+fort', **kwargs),
            GemmExample(**kwargs)]
