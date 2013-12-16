'''<b>MeasurementTemplate</b> - an example measurement module
<hr>
This is an example of a module that measures a property of an image both
for the image as a whole and for every object in the image. It demonstrates
how to load an image, how to load an object and how to record a measurement.

The text you see here will be displayed as the help for your module. You
can use HTML markup here and in the settings text; the Python HTML control
does not fully support the HTML specification, so you may have to experiment
to get it to display correctly.
'''
#
#
# Imports from useful Python libraries
#
#

import numpy as np
import scipy.ndimage as ndimage
import os
#
#
# Imports from CellProfiler
#
# The package aliases are the standard ones we use
# throughout the code.
#
#

import identify as cpmi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
# from cellprofiler.preferences import get_absolute_path, get_output_file_name
from cellprofiler.preferences import \
    DEFAULT_INPUT_FOLDER_NAME, \
    DEFAULT_OUTPUT_FOLDER_NAME, ABSOLUTE_FOLDER_NAME, \
    DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME
from cellprofiler.modules.loadimages import C_FILE_NAME, C_PATH_NAME

#
#
# Constants
#
# I put constants that are used more than once here.
#
# The Zernike list here is a list of the N & M Zernike
# numbers that we use below in our measurements. I made it
# a constant because the same list will be used to make
# the measurements and to report the measurements that can
# be made.
#
#
#
# The module class
#
# Your module should "inherit" from cellprofiler.cpmodule.CPModule.
# This means that your module will use the methods from CPModule unless
# you re-implement them. You can let CPModule do most of the work and
# implement only what you need.
#
#


class MeasurementSpots(cpm.CPModule):
    #
    #
    # The module starts by declaring the name that's used for display,
    # the category under which it is stored and the variable revision
    # number which can be used to provide backwards compatibility if
    # you add user-interface functionality later.
    #
    #
    module_name = "MeasurementSpots"
    category = "Measurement"
    variable_revision_number = 1

    #
    #
    # create_settings is where you declare the user interface elements
    # (the "settings") which the user will use to customize your module.
    #
    # You can look at other modules and in cellprofiler.settings for
    # settings you can use.
    #
    #

    def create_settings(self):
        #
        # The ImageNameSubscriber "subscribes" to all ImageNameProviders in
        # prior modules. Modules before yours will put images into CellProfiler
        # The ImageSubscriber gives your user a list of these images
        # which can then be used as inputs in your module.
        #

        self.input_image_name = cps.ImageNameSubscriber(
            # The text to the left of the edit box
            "Channel analized:",
            # HTML help that gets displayed when the user presses the
            # help button to the right of the edit box
            doc="""This needs to be the spot channel.
            """)

        self.input_image_coloc_name = cps.ImageNameSubscriber(
            # The text to the left of the edit box
            "Channel intensity:",
            # HTML help that gets displayed when the user presses the
            # help button to the right of the edit box
            doc="""This needs to be the intensity measured channel.
            """)

        self.input_object_mask = cps.ObjectNameSubscriber(
            "Cells:",
            doc="""Cell mask to get cell statistics.""")

        self.input_nuclei_mask = cps.ObjectNameSubscriber(
            "Nuclei:",
            doc="""Nuclei mask to get Nuclear to Cytoplasm ratio statistics.""")

        self.input_object_spots = cps.ObjectNameSubscriber(
            "Spots:",
            doc="""Spots binary mask extracted with SpotAnalizer""")

        self.directory = cps.DirectoryPath(
            "Output file location",
            dir_choices=[
                ABSOLUTE_FOLDER_NAME,
                DEFAULT_OUTPUT_FOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME,
                DEFAULT_INPUT_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME],
            doc="""
            This setting lets you choose the folder for the output
            files.""")

    #
    # The "settings" method tells CellProfiler about the settings you
    # have in your module. CellProfiler uses the list for saving
    # and restoring values for your module when it saves or loads a
    # pipeline file.
    #
    # This module does not have a "visible_settings" method. CellProfiler
    # will use "settings" to make the list of user-interface elements
    # that let the user configure the module. See imagetemplate.py for
    # a template for visible_settings that you can cut and paste here.
    #
    def settings(self):
        return [self.input_image_name,
                self.input_image_coloc_name,
                self.input_nuclei_mask,
                self.input_object_mask,
                self.input_object_spots]

    #
    # CellProfiler calls "run" on each image set in your pipeline.
    # This is where you do the real work.
    #
    def run(self, workspace):

        #
        input_mask_name = self.input_object_mask.value
        input_spot_name = self.input_object_spots.value
        input_intensity_name = self.input_image_coloc_name.value
        input_nuclei_name = self.input_nuclei_mask.value
        #
        # Get the measurements object - we put the measurements we
        # make in here
        #

        image_set = workspace.image_set
        input_image = image_set.get_image(input_intensity_name,
                                          must_be_grayscale=True)
        pixels = input_image.pixel_data

        meas = workspace.measurements
        object_set = workspace.object_set

        objects_input_mask = object_set.get_objects(input_mask_name)
        input_mask = objects_input_mask.segmented
        objects_input_spot = object_set.get_objects(input_spot_name)
        input_spot = objects_input_spot.segmented
        objects_input_nuclei = object_set.get_objects(input_nuclei_name)
        input_nuclei = objects_input_nuclei.segmented

        display_stats = [["cell", "number of spot"]]
        statistics = [["Mean", "Median", "SD"]]
        count_cells = []
        intensity_spots = []
        area_spots = []
        cytoplasm_intensity_mean = []
        cytoplasm_intensity_std = []
        nuclear_intensity_mean = []
        nuclear_intensity_std = []
        N2C = []
        unique_cells = np.unique(input_mask)

        #
        # Calculating Spots Statistics
        #
        IMG = (pixels * 65536).astype(np.int16)

        for i in unique_cells[1:]:
            # Getting the masks
            mask = (input_mask == i).astype(np.int)
            nuclei = (input_nuclei == i).astype(np.int)
            # Filtering IMG channel with Nuclei Mask and replacing 0 with nan
            nuclei_IMG = (IMG * nuclei).astype(np.float)
            cytoplasm_IMG = (IMG * mask).astype(np.float)
            for k in range(len(nuclei_IMG)):
                for j in range(len(nuclei_IMG[i])):
                    if nuclei_IMG[k][j] == 0:
                        nuclei_IMG[k][j] = np.nan
                    if cytoplasm_IMG[k][j] == 0:
                        cytoplasm_IMG[k][j] = np.nan
            # Calculating intensity
            nuclear_intensity_mean.append(np.nanmean(nuclei_IMG))
            nuclear_intensity_std.append(np.nanstd(nuclei_IMG))
            cytoplasm_intensity_mean.append(np.nanmean(cytoplasm_IMG))
            cytoplasm_intensity_std.append(np.nanstd(cytoplasm_IMG))
            # Nuclear to cytoplasm ratio
            N2C.append(nuclear_intensity_mean[-1] / cytoplasm_intensity_mean[-1])
            masked_spot = (input_spot * mask).astype(np.int16)
            lab, counts = ndimage.label(masked_spot)
            count_cells.append(counts)
            display_stats.append([i, counts])
            unique_spots = np.unique(lab)
            spot_intensity = []
            spot_area = []
            for i in unique_spots[1:]:
                spot_mask = (lab == i).astype(np.int16)
                intensity_masked = (IMG * spot_mask)
                tot_pixel = spot_mask.sum()
                tot_intensity = intensity_masked.sum()
                spot_intensity.append(tot_intensity)
                spot_area.append(tot_pixel)
            intensity_spots.append(spot_intensity)
            area_spots.append(spot_area)

        mean = np.mean(count_cells)
        median = np.median(count_cells)
        sd = np.std(count_cells)

        statistics.append([mean, median, sd])

        #
        # Calculating Cytoplasm and Nuclear FUS intensity
        #

        #
        # We record some statistics which we will display later.
        # We format them so that Matplotlib can display them in a table.
        # The first row is a header that tells what the fields are.
        #
        #
        # Put the statistics in the workspace display data so we
        # can get at them when we display
        #
        workspace.display_data.statistics = statistics
        workspace.display_data.counts = display_stats
        workspace.display_data.intensity_spots = intensity_spots
        workspace.display_data.area_spots = area_spots
        workspace.display_data.cytoplasm_intensity_mean = cytoplasm_intensity_mean
        workspace.display_data.cytoplasm_intensity_std = cytoplasm_intensity_std
        workspace.display_data.nuclear_intensity_mean = nuclear_intensity_mean
        workspace.display_data.nuclear_intensity_std = nuclear_intensity_std
        workspace.display_data.N2C = N2C
        cpmi.add_object_count_measurements(meas,
                                           input_spot_name, mean)
        for i in count_cells:
            workspace.add_measurement(input_spot_name, "Spots_CellCounts", [i])
        self.save(workspace)

    def is_interactive(self):
        return False

    def display(self, workspace, figure):
        statistics = workspace.display_data.statistics
        counts = workspace.display_data.counts
        figure.set_subplots((2, 1))
        figure.subplot_table(0, 0, statistics, ratio=(.25, .25, .25, .25))
        figure.subplot_table(1, 0, counts, ratio=(.25, .25, .25, .25))

    def save(self, workspace):
        # path = self.directory.value.replace(
            # "Default Input Folder sub-folder|", DEFAULT_INPUT_SUBFOLDER_NAME)
        measurements = workspace.measurements
        filename_feature = C_FILE_NAME + "_" + self.input_image_name.value
        pathname_feature = C_PATH_NAME + "_" + self.input_image_name.value
        # img_nb = measurements.image_number
        pathname = measurements.get_measurement(
            cpmeas.IMAGE, pathname_feature)
        filename = measurements.get_measurement(
            cpmeas.IMAGE, filename_feature)
        filename += '.spotdata'
        path = os.path.join(pathname, filename)

        f = open(path, 'w')
        for i in workspace.display_data.counts[1:]:
            f.write("Counts_%s\t%s\n" % (i[0], i[1]))
            f.write("Nuclear_Mean_Int_%s\t%s\n" % (i[0], workspace.display_data.nuclear_intensity_mean[i[0] - 1]))
            f.write("Nuclear_Mean_Std_%s\t%s\n" % (i[0], workspace.display_data.nuclear_intensity_std[i[0] - 1]))
            f.write("Cytoplasm_Mean_Int_%s\t%s\n" % (i[0], workspace.display_data.cytoplasm_intensity_mean[i[0] - 1]))
            f.write("Cytoplasm_Mean_Std_%s\t%s\n" % (i[0], workspace.display_data.cytoplasm_intensity_std[i[0] - 1]))
            f.write("N2C_%s\t%s\n" % (i[0], workspace.display_data.N2C[i[0] - 1]))
            tmp = "Intensity_%s" % i[0]
            for j in workspace.display_data.intensity_spots[i[0] - 1]:
                tmp += '\t%s' % j
            f.write(tmp + '\n')
            tmp = "Area_%s" % i[0]
            for j in workspace.display_data.area_spots[i[0] - 1]:
                tmp += '\t%s' % j
            f.write(tmp + '\n')

        f.close()
