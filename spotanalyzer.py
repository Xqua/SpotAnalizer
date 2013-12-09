'''<b>Automatic spot detection</b> - Extended morphological processing: a
practical method for automatic spot detection of biological markers from
microscopic images
<hr>
This is an implementation of the Extended morphological processing, published
by Yoshitaka Kimori, Norio Baba, and Nobuhiro Morone
in 2010 in BMC Bioinformatics.
doi:  10.1186/1471-2105-11-373

This module uses morphological transformation in order to automatically detects
spots in the picture. The spots can be from Fluoresence Microscopy images or
Electron Microscopy images.
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
from identify import draw_outline
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.objects as cpo
import cellprofiler.cpmath as cpmath
import cellprofiler.preferences as cpp
from cellprofiler.cpmath.filter import stretch


#
#
# Constants
#
# It's good programming practice to replace things like strings with
# constants if they will appear more than once in your program. That way,
# if someone wants to change the text, that text will change everywhere.
# Also, you can't misspell it by accident.
#
PP_MORPH = "Morphology Binary Opening"
PP_SIZE = "Size-based filter"
PP_GAUS = "Gaussian fitting"
PP_NONE = "No Post-Processing"

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


class SpotAnalyzer(cpm.CPModule):
    #
    #
    # The module starts by declaring the name that's used for display,
    # the category under which it is stored and the variable revision
    # number which can be used to provide backwards compatibility if
    # you add user-interface functionality later.
    #
    #
    module_name = "SpotAnalyzer"
    category = "Object Processing"
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
        # prior modules. Modules before yours will put images into
        # CellProfiler.
        # The ImageSubscriber gives your user a list of these images
        # which can then be used as inputs in your module.
        #
        self.input_image = cps.ImageNameSubscriber(
            "Input spot image:",
            doc="""This is the image that the module operates on. You can
            choose any image that is made available by a prior module.
            <br>
            <b>SpotAnalizer</b> will detect the spots on this image.
            """)

        self.output_spots = cps.ObjectNameProvider(
            "Output spot objects name:",
            "Spots",
            doc="""Enter the name that you want to call the objects identified
            by this module.""")

        self.apply_mask = cps.Binary(
            "Constrains the detection to some Objects ?",
            True,
            doc="""If selected, the spots detected outside of the
            input objects will be removed from the result.
            This is recomended as this will avoid noise detection outside
            of cell boundary.""")

        self.input_object_mask = cps.ObjectNameSubscriber(
            "Constrained Objects",
            doc="""Objects to serve as a restriction mask.""")

        self.angles = cps.Integer(
            "Number of performed rotations:",
            36,
            minval=1,
            maxval=180,
            doc="""This parameter correspond to the number of rotations
            performed for the rotational morphological processing.
            According to the authors, 36 is the best TradeOff.
            The higher the number is the better the accuracy.
            But each rotation increase the computational cost.""")

        self.SE_min = cps.Integer(
            "Minimum Spot size:",
            2,
            minval=1,
            maxval=100,
            doc="""This parameter should be set to be smaler than the
            smalest pixel size of the spots you are extracting.""")

        self.SE_max = cps.Integer(
            "Maximum Spot size:",
            7,
            minval=1,
            maxval=100,
            doc="""This parameter should be set to be bigger than the
            biggest pixel size of the spots you are extracting.""")

        self.Threshold = cps.Integer(
            "Threshold:",
            0,
            minval=0,
            maxval=100,
            doc="""This parameter corespond to the Threshold used to
            select the spots after the morphological transformations
            were executed. It should be set to 0 to be less stringent,
            and increased if too many False Discovery happens. By
            increasing the value, you are reducing the False Discovery
            Rate, but this may reduce the True Discovery Rate too.""")

        self.post_processing = cps.Choice(
            'Method to distinguish clumped objects',
            [PP_NONE, PP_MORPH, PP_SIZE, PP_GAUS],
            doc="""Post Processing can be useful in order to remove
            persisting noise, or false postive.<br>
            <ul>
            <li>%(PP_MORPH)%: Method based on morphology, remove noise using
            the morphology opening function.</li>
            <li>%(PP_SIZE)%: Method based on the size of the objects. Any
            object not in the size range will be removed.</li>
            <li>%(PP_NONE)%: No Post-Processing.</li>
            <li>%(PP_GAUS)%: Each Object is fitted with a gaussian.</li>
            </ul>""" % globals())

        self.gaussian_threshold = cps.Float(
            'Correlation Threshold',
            0.8,
            minval=0,
            maxval=1,
            doc="""This value correspond to the Threshold to be applied on the
            gaussian fit. When an object is fitted with a 2G gaussian
            distribution, it's correlation is extracted. If the correlation
            is inferior to the Threshold the spot is removed.""")

        self.size_range = cps.IntegerRange(
            "Allowed Spot size",
            (2, 7), minval=1, doc='''
            This setting correspond to the range size of allowed spots.
            It can be useful to remove too small or too big wrongly
            detected spots''')

    #
    # The "settings" method tells CellProfiler about the settings you
    # have in your module. CellProfiler uses the list for saving
    # and restoring values for your module when it saves or loads a
    # pipeline file.
    #
    def settings(self):
        return [self.input_image, self.output_spots, self.apply_mask,
                self.input_object_mask, self.angles, self.SE_max,
                self.SE_min, self.Threshold, self.post_processing,
                self.size_range]

    #
    # visible_settings tells CellProfiler which settings should be
    # displayed and in what order.
    #
    # You don't have to implement "visible_settings" - if you delete
    # visible_settings, CellProfiler will use "settings" to pick settings
    # for display.
    #
    def visible_settings(self):
        result = [self.input_image, self.output_spots, self.apply_mask]
        #
        # Show the user the scale only if self.wants_smoothing is checked
        #
        if self.apply_mask:
            result += [self.input_object_mask]
        result += [self.angles, self.SE_max, self.SE_min, self.Threshold,
                   self.post_processing]
        if self.post_processing.value == PP_SIZE:
            result += [self.size_range]
        return result

    #
    # CellProfiler calls "run" on each image set in your pipeline.
    # This is where you do the real work.
    #
    def run(self, workspace):
        #
        # Get the input and output image names. You need to get the .value
        # because otherwise you'll get the setting object instead of
        # the string name.
        #
        input_image_name = self.input_image.value
        if self.apply_mask:
            input_mask_name = self.input_object_mask.value
        # output_spot_name = self.output_spots.value
        #
        # Get the image set. The image set has all of the images in it.
        # The assert statement makes sure that it really is an image set,
        # but, more importantly, it lets my editor do context-sensitive
        # completion for the image set.
        #
        image_set = workspace.image_set
        # assert isinstance(image_set, cpi.ImageSet)
        #
        # Get the input image object. We want a grayscale image here.
        # The image set will convert a color image to a grayscale one
        # and warn the user.
        #
        input_image = image_set.get_image(input_image_name,
                                          must_be_grayscale=True)
        #
        # Get the pixels - these are a 2-d Numpy array.
        #
        image_pixels = input_image.pixel_data

        # Getting the mask
        if self.apply_mask:
            object_set = workspace.object_set
            # assert isinstance(object_set, cpo.ObjectSet)
            objects = object_set.get_objects(input_mask_name)
            input_mask = objects.segmented
        else:
            input_mask = None

        #
        # Get the smoothing parameter
        #
        spots = self.Spot_Extraction(
            image_pixels,
            mask=input_mask,
            N=self.angles.value,
            l_noise=self.SE_min.value,
            l_spot=self.SE_max.value,
            Threshold=self.Threshold.value)

        labeled_spots, counts_spots = ndimage.label(spots)
        #
        # Post Processing
        #
        if self.post_processing.value == PP_MORPH:
            labeled_spots_filtered = self.filter_on_morph(labeled_spots)
        elif self.post_processing.value == PP_SIZE:
            labeled_spots, labeled_spots_filtered = self.filter_on_size(
                labeled_spots, counts_spots)
        elif self.post_processing.value == PP_GAUS:
            labeled_spots_filtered = self.filter_on_gaussian(image_pixels,
                                                             labeled_spots)
        else:
            labeled_spots_filtered = labeled_spots
        labeled_spots_filtered, counts_spots_filtered = ndimage.label(
            labeled_spots_filtered)
        #
        # Make an image object. It's nice if you tell CellProfiler
        # about the parent image - the child inherits the parent's
        # cropping and masking, but it's not absolutely necessary
        #
        # Add image measurements
        objname = self.output_spots.value
        measurements = workspace.measurements
        cpmi.add_object_count_measurements(measurements,
                                           objname, counts_spots_filtered)
        # Add label matrices to the object set
        objects = cpo.Objects()
        objects.segmented = labeled_spots_filtered
        objects.unedited_segmented = labeled_spots
        objects.post_processed = labeled_spots_filtered
        objects.parent_image = input_image

        outline_image = cpmath.outline.outline(labeled_spots)
        outline_image_filtered = cpmath.outline.outline(labeled_spots_filtered)

        workspace.object_set.add_objects(objects, self.output_spots.value)
        cpmi.add_object_location_measurements(workspace.measurements,
                                              self.output_spots.value,
                                              labeled_spots_filtered)
        #
        # Save intermediate results for display if the window frame is on
        #
        workspace.display_data.input_pixels = image_pixels
        workspace.display_data.output_pixels = spots
        workspace.display_data.outline_image = outline_image
        workspace.display_data.outline_image_filtered = outline_image_filtered

    def Spot_Extraction(self, IMG, mask=None, N=36, l_noise=3, l_spot=6,
                        Threshold=0, Noise_reduction=True):
        """N : Number of rotations to perform
        l_noise : lenght of straigh line segment for Strucure element (must be
        < to the spot size)
        l_spot : lenght of SE for spot extraction (must be > to the spot size)
        Threshold : Threshold value for spot selection"""
        IMG = (IMG * 65536).astype(np.int32)
        IMG_Noise_red = self.RMP(IMG, N, l_noise)
        IMG_Spot = self.RMP(IMG_Noise_red, N, l_spot)
        IMG_TopHat = IMG - IMG_Spot
        IMG_Spots = (IMG_TopHat > Threshold).astype(np.int)
        if mask is not None:
            IMG_Spots = IMG_Spots * mask
        return IMG_Spots

    def RMP(self, IMG, N, l):
        opened_images = []
        for n in range(N):
            angle = (180 / N) * n
            IMG_rotate = ndimage.interpolation.rotate(
                IMG, angle, reshape=True, mode="constant", cval=0)
            IMG_Opened = ndimage.morphology.grey_opening(
                IMG_rotate, size=(0, l))
            IMG2 = ndimage.interpolation.rotate(
                IMG_Opened, -angle, reshape=True, mode="constant", cval=0)
            a = IMG.shape
            b = IMG2.shape
            if a != b:
                x = (b[0] - a[0]) / 2
                y = (b[1] - a[1]) / 2
                IMG2 = IMG2[x:x + a[0], y:y + a[1]]
            opened_images.append(IMG2)
        return np.array(opened_images, dtype=np.int32).max(axis=0)

    def filter_on_morph(self, IMG):
        """Filter the spot image based the morphology opening function."""
        return ndimage.morphology.binary_opening(IMG)

    def filter_on_gaussian(self, IMG, spots):
        "Filter the spots based on gaussian correlation."
        return spots

    def filter_on_size(self, labeled_image, object_count):
        """ Filter the labeled image based on the size range
        labeled_image - pixel image labels
        object_count - # of objects in the labeled image
        returns the labeled image, and the labeled image with the
        small objects removed
        """
        if object_count > 0:
            areas = ndimage.measurements.sum(np.ones(labeled_image.shape),
                                             labeled_image,
                                             np.array(range(0,
                                                            object_count + 1),
                                                      dtype=np.int32))
            areas = np.array(areas, dtype=int)
            min_allowed_area = np.pi * \
                (self.size_range.min * self.size_range.min) / 4
            max_allowed_area = np.pi * \
                (self.size_range.max * self.size_range.max) / 4
            # area_image has the area of the object at every pixel within the
            # object
            area_image = areas[labeled_image]
            labeled_image[area_image < min_allowed_area] = 0
            small_removed_labels = labeled_image.copy()
            labeled_image[area_image > max_allowed_area] = 0
        else:
            small_removed_labels = labeled_image.copy()
        return (labeled_image, small_removed_labels)

    #
    # is_interactive tells CellProfiler whether "run" uses any interactive
    # GUI elements. If you return False here, CellProfiler will run your
    # module on a separate thread which will make the user interface more
    # responsive.
    #
    def is_interactive(self):
        return False
    #
    # display lets you use matplotlib to display your results.
    #

    def display(self, workspace, figure):
        #
        # the "figure" is really the frame around the figure. You almost always
        # use figure.subplot or figure.subplot_imshow to get axes to draw on
        # so we pretty much ignore the figure.
        #
        # figure = workspace.create_or_find_figure(subplots=(2, 1))
        #
        figure.set_subplots((2, 1))
        # Show the user the input image
        #
        orig_axes = figure.subplot(0, 0)
        figure.subplot(1, 0, sharexy=orig_axes)

        figure.subplot_imshow_grayscale(
            0, 0,  # show the image in the first row and column
            workspace.display_data.input_pixels,
            title=self.input_image.value)
        #
        # Show the user the final image
        #

        if workspace.display_data.input_pixels.ndim == 2:
            # Outline the size-excluded pixels in red
            outline_img = np.ndarray(
                shape=(workspace.display_data.input_pixels.shape[0],
                       workspace.display_data.input_pixels.shape[1], 3))
            outline_img[:, :, 0] = workspace.display_data.input_pixels
            outline_img[:, :, 1] = workspace.display_data.input_pixels
            outline_img[:, :, 2] = workspace.display_data.input_pixels
        else:
            outline_img = workspace.display_data.image.copy()
        #
        # Stretch the outline image to the full scale
        #
        outline_img = stretch(outline_img)

        # Outline the accepted objects pixels
        draw_outline(outline_img, workspace.display_data.outline_image,
                     cpp.get_secondary_outline_color())

        # Outline the size-excluded pixels
        draw_outline(outline_img,
                     workspace.display_data.outline_image_filtered,
                     cpp.get_primary_outline_color())

        title = "%s outlines" % (self.output_spots.value)
        figure.subplot_imshow(1, 0, outline_img, title, normalize=False)
