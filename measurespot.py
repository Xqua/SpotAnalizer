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
import cellprofiler.settings as cps
from cellprofiler.preferences import \
    DEFAULT_INPUT_FOLDER_NAME, \
    DEFAULT_OUTPUT_FOLDER_NAME, ABSOLUTE_FOLDER_NAME, \
    DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME

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

        self.input_object_mask = cps.ObjectNameSubscriber(
            "Cells:",
            doc="""Cell mask to get cell statistics.""")

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
        return [self.input_object_mask,
                self.input_object_spots,
                self.directory]

    #
    # CellProfiler calls "run" on each image set in your pipeline.
    # This is where you do the real work.
    #
    def run(self, workspace):

        #
        input_mask_name = self.input_object_mask.value
        input_spot_name = self.input_object_spots.value

        #
        # Get the measurements object - we put the measurements we
        # make in here
        #
        meas = workspace.measurements
        object_set = workspace.object_set

        objects_input_mask = object_set.get_objects(input_mask_name)
        input_mask = objects_input_mask.segmented
        objects_input_spot = object_set.get_objects(input_spot_name)
        input_spot = objects_input_spot.segmented

        display_stats = [["cell", "number of spot"]]
        statistics = [["Mean", "Median", "SD"]]
        count_cells = []
        unique_cells = np.unique(input_mask)

        for i in unique_cells[1:]:
            mask = (input_mask == i).astype(np.int)
            masked_spot = input_spot * mask
            lab, counts = ndimage.label(masked_spot)
            count_cells.append(counts)
            display_stats.append([i, counts])

        mean = np.mean(count_cells)
        median = np.median(count_cells)
        sd = np.std(count_cells)

        statistics.append([mean, median, sd])
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
        cpmi.add_object_count_measurements(meas,
                                           input_spot_name, mean)
        meas.add_measurement(input_spot_name, "Spot_counts", count_cells)
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
        path = self.directory.value.replace(
            "Default Input Folder sub-folder|", "/home/xqua/")
        f = open(path + '/cell_count.tsv', 'a')
        # f.write(">cell\tcount\n")
        for i in workspace.display_data.counts:
            f.write("%s\t%s\n" % (i[0], i[1]))
        f.close()