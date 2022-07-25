import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from turb2d import TurbidityCurrent2D
from turb2d.utils import create_topography, create_init_flow_region
import numpy as np
from contextlib import contextmanager
import signal
import sys
import warnings
from tqdm import tqdm

import time
import multiprocessing as mp
import netCDF4
from landlab.io.native_landlab import save_grid


class RunMultiFlows():
    """A class to run multiple flows for conducting inverse analysis
    """
    def __init__(
            self,
            C_ini,
            r_ini,
            h_ini,
            dirpath,
            filename,
            grain_class_num,
            processors=1,
            endtime=1000,
            timelimit=None
    ):

        self.C_ini = C_ini
        self.r_ini = r_ini
        self.h_ini = h_ini
        self.filename = filename
        self.dirpath = dirpath
        self.num_runs = len(C_ini)
        self.processors = processors
        self.endtime = endtime
        self.timelimit = timelimit
        self.grain_class_num = grain_class_num
        # self.num_runs = C_ini.shape[0]

    def produce_flow(self, C_ini, r_ini, h_ini):
        """ producing a TurbidityCurrent2D object.
        """

        # create a grid
        grid = create_topography(
            length=5000,
            width=2000,
            spacing=10,
            slope_outside=0.2,
            slope_inside=0.05,
            slope_basin_break=2000,
            canyon_basin_break=2200,
            canyon_center=1000,
            canyon_half_width=100,
        )

        create_init_flow_region(
            grid,
            initial_flow_concentration=C_ini,
            initial_flow_thickness=h_ini,
            initial_region_radius=r_ini,
            initial_region_center=[1000, 4000],
        )

        # making turbidity current object
        tc = TurbidityCurrent2D(grid,
                                Cf=0.004,
                                alpha=0.4,
                                kappa=0.05,
                                Ds=[3.5 *10**(-4),1.8 * 10**(-4), 8.8 * 10**(-5), 4.4 * 10**(-5)],
                                h_init=0.00001,
                                h_w=0.01,
                                C_init=0.00001,
                                implicit_num=100,
                                implicit_threshold=1.0 * 10**-5,
                                r0=1.5,
                                model='4eq')

        return tc
# Ds=[3.5 *10**(-4),1.8 * 10**(-4), 8.8 * 10**(-5), 4.4 * 10**(-5)]

    def run_flow(self, init_values):
        """ Run a flow to obtain the objective function
        """

        grain_class_num = self.grain_class_num
        # Produce flow object
        tc = self.produce_flow(init_values[1], init_values[2], init_values[3])
        #init_values=init_value_list->init_value[1]=init_values_list[1]-> init_values[1] is not C_ini?

        # Run the model until endtime or 99% sediment settled
        Ch_init = np.sum(tc.Ch)
        t = 0
        dt = 20

        # last=200
        # try:
        #     with self.timeout(self.timelimit):
        #         for j in range(1, last + 1):
        #             try:
        #                 tc.run_one_step(dt=dt)
        #                 if np.sum(tc.C * tc.h) / Ch_init < 0.005:
        #                     break
        #             except RuntimeWarning as e:
        #                 with open('RuntimeWarning.txt', mode='a+') as f:
        #                     f.write('{} \n'.format(init_values[0]))
        #                 print("Run no. {} RuntimeWarning".format(init_values[0]))
        #                 sys.exit(1)
        # except self.TimeoutException as e:
        #     with open('timeout.txt', mode='a+') as f:
        #         f.write('{} \n'.format(init_values[0]))
        #     print("Run no. {} timed out: {}".format(init_values[0],self.timelimit))

        if self.timelimit is None:
            while (((np.sum(tc.Ch) / Ch_init) > 0.01) and (t < self.endtime)):
                tc.run_one_step(dt=dt)
                t += dt

        elif type(self.timelimit) is int:
            try:
                # I used sytax "with" because it can guarantee that critical post-processing 
                # (code block limited by 'finally') will always be performed. 
                with self.timeout(self.timelimit):
                    while (((np.sum(tc.Ch) / Ch_init) > 0.01) and (t < self.endtime)):
                        # print(t)
                        try:
                            tc.run_one_step(dt=dt)
                            t += dt
                        except RuntimeWarning as e:
                            print("runtimewarning!")
                            with open('RuntimeWarning.txt', mode='a') as f:
                                f.write('{} \n'.format(init_values[0]))
                            print("Run no. {} RuntimeWarning".format(init_values[0]))
                            sys.exit(1)
                # print("aaaaaa")
            except self.TimeoutException as e:
                # print("qwerty")
                with open(os.path.join(self.dirpath,'timeout.txt'), mode='a') as f:
                    f.write('{}\n'.format(init_values[0]))
                print('Run no.{} timed out:{}'.format(init_values[0],self.timelimit))
    
        else:
            raise ValueError('An invalid value was entered in timelimit.')

        # save_grid(
        #     tc.grid,
        #     'run-{0:.3f}-{1:.3f}-{2:.3f}.grid'.format(
        #         init_values[0], init_values[1], init_values[2]),
        #     clobber=True)

        bed_thick = tc.grid.node_vector_to_raster(
            tc.grid.at_node['bed__thickness'])

        sed_volume_per_unit_area = []
        for i in range(grain_class_num):
            sed_volume_per_unit_area.append(
            tc.grid.node_vector_to_raster(
            tc.grid.at_node['bed__sediment_volume_per_unit_area_{}'.format(i)]    
            )
        )

        self.save_data(init_values, bed_thick, sed_volume_per_unit_area)

        print('Run no. {} finished'.format(init_values[0]))

    def save_data(self, init_values, bed_thick_i, sed_volume_per_unit_area_i):
        """Save result to a data file.
        """

        grain_class_num = self.grain_class_num

        run_id = init_values[0]
        C_ini_i = init_values[1]
        r_ini_i = init_values[2]
        h_ini_i = init_values[3]

        dfile = netCDF4.Dataset(self.filename, 'a', share=True)
        C_ini = dfile.variables['C_ini']
        r_ini = dfile.variables['r_ini']
        h_ini = dfile.variables['h_ini']
        bed_thick = dfile.variables['bed_thick']

        C_ini[run_id] = C_ini_i
        r_ini[run_id] = r_ini_i
        h_ini[run_id] = h_ini_i
        bed_thick[run_id, :, :] = bed_thick_i
        for i in range(grain_class_num):
            sed_volume_per_unit_area = dfile.variables['sed_volume_per_unit_area_{}'.format(i)]
            sed_volume_per_unit_area[run_id, :, :] = sed_volume_per_unit_area_i[i]

        dfile.close()

    def run_multiple_flows(self):
        """run multiple flows
        """

        C_ini = self.C_ini
        r_ini = self.r_ini
        h_ini = self.h_ini

        # Create list of initial values
        init_value_list = list()
        for i in range(len(C_ini)):
            init_value_list.append([i, C_ini[i], r_ini[i], h_ini[i]])

        # run flows using multiple processors
        pool = mp.Pool(self.processors)
        pool.map(self.run_flow, init_value_list)
        pool.close()
        pool.join()

    def create_datafile(self):

        num_runs = self.num_runs
        grain_class_num = self.grain_class_num

        # check grid size
        C_list = np.full(grain_class_num, 0.01)
        tc = self.produce_flow(C_list, 100, 100)
        grid_x = tc.grid.nodes.shape[0]
        grid_y = tc.grid.nodes.shape[1]
        dx = tc.grid.dx

        # record dataset in a netCDF4 file
        datafile = netCDF4.Dataset(self.filename, 'w')
        datafile.createDimension('run_no', num_runs)
        datafile.createDimension('grid_x', grid_x)
        datafile.createDimension('grid_y', grid_y)
        datafile.createDimension('basic_setting', 1)
        datafile.createDimension('num_gs', grain_class_num)

        spacing = datafile.createVariable('spacing',
                                          np.dtype('float64').char,
                                          ('basic_setting'))
        spacing.long_name = 'Grid spacing'
        spacing.units = 'm'
        spacing[0] = dx

        C_ini = datafile.createVariable('C_ini',
                                        np.dtype('float64').char, ('run_no', 'num_gs'))
        C_ini.long_name = 'Initial Concentration'
        C_ini.units = 'Volumetric concentration (dimensionless)'
        r_ini = datafile.createVariable('r_ini',
                                        np.dtype('float64').char, ('run_no'))
        r_ini.long_name = 'Initial Radius'
        r_ini.units = 'm'
        h_ini = datafile.createVariable('h_ini',
                                        np.dtype('float64').char, ('run_no'))
        h_ini.long_name = 'Initial Height'
        h_ini.units = 'm'

        bed_thick = datafile.createVariable('bed_thick',
                                            np.dtype('float64').char,
                                            ('run_no', 'grid_x', 'grid_y'))
        bed_thick.long_name = 'Bed thickness'
        bed_thick.units = 'm'

        for i in range(grain_class_num):
            sed_volume_per_unit_area = datafile.createVariable('sed_volume_per_unit_area_{}'.format(i),
                                                np.dtype('float64').char,
                                                ('run_no', 'grid_x', 'grid_y'))
            sed_volume_per_unit_area.long_name = 'sediment_volume_per_unit_area_{}'.format(i)
            sed_volume_per_unit_area.units = 'm'

        # close dateset
        datafile.close()

    # definition of class for timeout    
    class TimeoutException(Exception):
        pass

    @contextmanager
    def timeout(self, seconds):
        def handler(signum, frame):
            raise self.TimeoutException('Time out')
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            # create contextmanager
            yield
        finally:
            # task when end of contextmanager
            # initialization of alarm setting for signal.SIGALRM
            signal.alarm(0)


if __name__ == "__main__":
    # ipdb.set_trace()

    proc = 10  # number of processors to be used
    num_runs = 300
    Cmin, Cmax = [0.001, 0.03]
    rmin, rmax = [50., 200.]
    hmin, hmax = [25., 150.]

    C_ini = np.random.uniform(Cmin, Cmax, num_runs)
    r_ini = np.random.uniform(rmin, rmax, num_runs)
    h_ini = np.random.uniform(hmin, hmax, num_runs)

    rmf = RunMultiFlows(
        C_ini,
        r_ini,
        h_ini,
        'super191208_01.nc',
        processors=proc,
        endtime=4000.0,
    )
    rmf.create_datafile()
    start = time.time()
    rmf.run_multiple_flows()
    print("elapsed time: {} sec.".format(time.time() - start))
