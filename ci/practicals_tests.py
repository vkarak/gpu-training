import os

import reframe.utility.sanity as sn
from reframe.core.modules import get_modules_system
from reframe.core.pipeline import RegressionTest


class OpenACCBaseTest(RegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['dom:gpu', 'daint:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.sourcesdir = os.path.join(self.prefix, '../solutions')
        self.modules = ['craype-accel-nvidia60']
        self.maintainers = ['karakasis<at>cscs.ch']

    # Remove underscore to test with PGI 18.4 (but use -p PrgEnv-pgi only!)
    def _setup(self, partition, environ, **job_opts):
        if environ.name == 'PrgEnv-pgi':
            get_modules_system().searchpath_add(
                '/apps/common/UES/pgi/18.4/modulefiles')
            self.modules += ['pgi/18.4']
            self.variables.update({'PGI_VERS_STR': '18.4.0'})
            self.pre_run = ['module use /apps/common/UES/pgi/18.4/modulefiles']

        super().setup(partition, environ, **job_opts)

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
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.variables = {
                'CRAY_CUDA_MPS': '1',
                'MPICH_RDMA_ENABLED_CUDA': '1'
            }

        self.sanity_patterns = sn.assert_found('writing to output',
                                               self.stdout)
        self.keep_files = ['output.bin', 'output.bov']


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


@sn.sanity_function
def dset(iterable):
    return set(iterable)


class ImagePipelineExample(OpenACCBaseTest):
    def __init__(self, **kwargs):
        super().__init__('image_pipeline_example', **kwargs)
        self.sourcepath = 'image-pipeline/'
        self.valid_prog_environs = ['PrgEnv-pgi']

        # We need to reload the PGI compiler here, cos OpenCV loads PrgEnv-gnu
        self.modules = ['craype-accel-nvidia60', 'OpenCV', 'pgi']
        self.executable = './image-pipeline/filter.x'
        self.executable_opts = ['image-pipeline/california-1751455_1280.jpg',
                                'image-pipeline/output.jpg']
        self.sanity_patterns = sn.assert_eq(
            {'original', 'blocked', 'update', 'pipelined', 'multi'},
            dset(sn.extractall('Time \((\S+)\):.*', self.stdout, 1)))


class DeepcopyExample(OpenACCBaseTest):
    def __init__(self, version, **kwargs):
        super().__init__('deepcopy_%s_example' % version.replace('+', '_'),
                         **kwargs)
        self.sourcepath = 'deepcopy/'
        self.valid_prog_environs = ['PrgEnv-pgi']
        self.modules = ['craype-accel-nvidia60']
        self.executable = './deepcopy/deepcopy.%s' % version.replace('+', '.')
        self.sanity_patterns = sn.assert_found('3', self.stdout)


def _get_checks(**kwargs):
    return [AXPYExample(**kwargs),
            AXPYExample('fortran', **kwargs),
            DiffusionExample('omp', **kwargs),
            BlurExample('openacc', **kwargs),
            BlurExample('openacc+fort', **kwargs),
            DeepcopyExample('openacc', **kwargs),
            DeepcopyExample('openacc+fort', **kwargs),
            DiffusionExample('openacc', **kwargs),
            DiffusionExample('openacc+cuda', **kwargs),
            DiffusionExample('openacc+cuda+mpi', **kwargs),
            DiffusionExample('openacc+mpi', **kwargs),
            DiffusionExample('openacc+fort', **kwargs),
            DiffusionExample('openacc+fort+mpi', **kwargs),
            DotExample('openacc', **kwargs),
            DotExample('openacc+fort', **kwargs),
            GemmExample(**kwargs),
            ImagePipelineExample(**kwargs)]
