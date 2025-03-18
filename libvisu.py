# This script generates a video from a sequence of pictures. Use the Imaging conda env to run.
import pandas as pd
import sys, cv2, os
sys.path.append("ABCImaging/VideoManagment")
sys.path.append("ABCImaging/Preprocessing")
sys.path.append("ABCThermalPlots")
from io import StringIO
from videolib import *
from thermalutil import *
from preproc import beautify_frame
from PIL import Image  # Or OpenCV if preferred
from matplotlib.path import Path

def fetchImagesPaths(rootpath_imgs, datetimes, hive_nb, images_fill_limit = 30):
    '''
    Fetches the images' paths for a specific hive at specific datetimes.
    Parameters:
    - rootpath_imgs: str, root path to the images
    - datetimes: pd.DatetimeIndex, datetimes for which we want the images
    - hive: int, hive number
    - images_fill_limit: int, maximum number of images to fill the gaps with the previous images. Default is 30 (5 hours).
    Returns:
    - imgs_paths_filtered: pd.DataFrame, containing the image paths. Each row is a datetime, each column is a RPi.
    '''

    # Get the list of folders in the rootpath
    paths = [os.path.join(rootpath_imgs, f) for f in os.listdir(rootpath_imgs) if os.path.isdir(os.path.join(rootpath_imgs, f))]
    paths = [path for path in paths if "h"+hive_nb in path]
    rpis = [path.split("/")[-1][3] for path in paths]

    # Order the paths alphabetically
    paths.sort() # Now this contains the path to all RPis images

    # For each dt in datetimes, find the image path that == dt for each RPi. Put the paths in a df where each row is a dt and each column is a RPi
    imgs_paths = pd.DataFrame(index=datetimes, columns=[os.path.basename(path)[:4] for path in paths])
    for dt in datetimes:
        for path in paths:
            filename = "hive"+hive_nb+"_rpi"+path.split("/")[-1][3]+"_"+dt.strftime('%y%m%d-%H%M')
            # Find the file in os.listdir(path) that contains the dt (or startswith(dt))
            img_path = [os.path.join(path, f) for f in os.listdir(path) if filename in f]
            if len(img_path) == 1:
                imgs_paths.loc[dt, os.path.basename(path)[:4]] = img_path[0]
            else:
                imgs_paths.loc[dt, os.path.basename(path)[:4]] = None

    # Check how many images are missing
    print("Missing images before filtering: ", imgs_paths.isnull().sum().sum(), "out of", imgs_paths.shape[0]*imgs_paths.shape[1], "images.")

    # Fill the gaps with the images from the previous dt
    imgs_paths_filtered = imgs_paths.ffill(limit=images_fill_limit,axis=0)
    # Check if there are still missing images, if so, raise an error
    if imgs_paths_filtered.isnull().sum().sum() > 0:
        raise ValueError(f"Still missing images, desipite filling the gaps with the previous images up to {images_fill_limit} images.")
    
    return imgs_paths_filtered

# Function to access files and force download
def preload_images(src_path):
    for root, dirs, files in os.walk(src_path):
        for file in files:
            # Check for image file extensions
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)
                try:
                    # Attempt to open the file
                    with Image.open(file_path) as img:
                        img.verify()  # Ensure file integrity
                    print(f"Preloaded: {file_path}")
                except Exception as e:
                    print(f"Failed to preload {file_path}: {e}")

def prepareData(csv_path, section_prefix="#"):
    """
    Reads a CSV file split into multiple sections separated by lines starting with a specific prefix.
    Each section begins with `num_header_lines` lines starting with `section_prefix`, 
    and the first line after those is treated as column names.

    Args:
        file_path (str): Path to the CSV file.
        section_prefix (str): Prefix marking the start of a header line (e.g., "#").
        num_header_lines (int): Number of lines starting with `section_prefix` to mark the end of a section delimiter.

    Returns:
        all_data (pd.DataFrame): DataFrame containing all data from the CSV file.
    """
    with open(csv_path, 'r') as file:
        lines = file.readlines()
    
    # Variables to store sections
    sections = {}
    current_section = 0
    current_data = []
    column_names = None
    header_line_count = 0
    
    for line in lines:
        line = line.strip()
        
        if line.startswith(section_prefix):  # Count header lines starting with `#`
            header_line_count += 1
        elif header_line_count > 0:  # Just completed section header
            # Save the previous section
            if current_data!=[]:
                csv_data = '\n'.join(current_data)
                sections[current_section] = pd.read_csv(StringIO(csv_data), names=column_names)
            
            # Start a new section
            # The current line contains the column names as it just succeeded the header lines
            column_names = line.split(",")  # Use the delimiter to get the column names
            header_line_count = 0  # Mark header handling complete
            current_section = current_section+1  # Increment section name
            current_data = []  # Reset the data list
        elif column_names is None:
            continue  # Skip any lines before column names are defined
        else:
            # Add data lines to the current section
            current_data.append(line)
    
    # Save the last section
    if current_data:
        csv_data = '\n'.join(current_data)
        sections[current_section] = pd.read_csv(StringIO(csv_data), names=column_names)

    all_data = pd.concat(sections.values(), ignore_index=True, sort=False)
    all_data.drop(columns=['result', '_start', '_stop', 'table', 'serial_id', 'geo_loc', 'phys_loc'], inplace=True) # Remove unnecessary columns
    all_data['_time'] = all_data['_time'].str.replace(r'\.\d+', '', regex=True) # Remove milliseconds
    all_data['_time'] = pd.to_datetime(all_data['_time'], format="%Y-%m-%d %H:%M:%S%z")

    # Set the time as index
    all_data = all_data.set_index('_time',drop=False)
    # Order by time
    all_data = all_data.sort_index()
    return all_data
    
def extractData(df:pd.DataFrame, hive:int, timestamps:pd.DatetimeIndex):
    '''
    Extracts data from a dataframe for a specific hive and specific timestamps.
    '''
    relevant_data = df[(df['hive_num'] == hive) & (df.index.isin(timestamps))]
    if relevant_data.empty:
        print(f"No data found between for hive {hive} at the timestamps provided!")
        return None
    return relevant_data

def generateThermalDF(df:pd.DataFrame, index:pd.DatetimeIndex)->pd.DataFrame:
    '''
    Generates a panda df for temperatures that is friendly with the thermalutil.py libray. This means every column is a t00-t64 and a line is a timestamp.
    Parameters:
    - df: pd.DataFrame containing the temperatures in an influxdb format.
    '''
    upper = pd.DataFrame(index=index)
    lower = pd.DataFrame(index=index)
    df_upper = df[(df['_measurement'] == 'tmp') & (df['inhive_loc'] == 'upper')]
    df_lower = df[(df['_measurement'] == 'tmp') & (df['inhive_loc'] == 'lower')]

    for i in range(64):
        column_name = f't{i:02d}'
        # For every index of thermal_df, get the temperature which has the same datetime in df
        upper[column_name] = df_upper[df_upper['_field'] == column_name]['_value']
        lower[column_name] = df_lower[df_lower['_field'] == column_name]['_value']

    # Set the right column names
    upper.columns = [f't{i:02d}' for i in range(64)]
    lower.columns = [f't{i:02d}' for i in range(64)]
    return upper, lower

def generateMetabolicDF(df:pd.DataFrame)->pd.DataFrame:
    '''
    Generates a df for the metabolic data. Every column is a metabolic measure (ul,ur,ll,lr) and every line is a timestamp.
    parameters:
    - df: pd.DataFrame containing all data for our hive and timestamps. Index should be the timestamps.
    '''
    metabolic_out = pd.DataFrame(index=df.index)
    # Remove duplicates in index
    metabolic_out = metabolic_out[~metabolic_out.index.duplicated(keep='first')]
    metabolic_in = df[(df['_measurement'] == 'co2') & (df['_field'] == 'co2')]

    column_names = ['ul','ur','ll','lr'] # Upper left, upper right, lower left, lower right
    for column_name in column_names:
        inhive_loc = "upper" if column_name[0] == "u" else "lower"
        if column_name == "ul":
            rpi_num = 1
        elif column_name == "ll":
            rpi_num = 2
        else:
            rpi_num = 3

        metabolic_out[column_name] = metabolic_in[(metabolic_in['inhive_loc'] == inhive_loc) & (metabolic_in['rpi_num'] == rpi_num)]['_value']

    return metabolic_out

def generateHtrDF(df:pd.DataFrame)->list:
    '''
    Generates a list of df for the heater data. Lines of the df are the (same) timestamps, and columns are:
    - status
    - pwm
    - avg_temp
    - obj
    - actuator_instance

    We should thus have 10 rows in the df, and as many items in the list as there are timestamps in df.
    '''
    htr_in = df[(df['_measurement'] == 'htr')]
    timestamps = htr_in.index.unique()
    upper_out = []
    lower_out = []
    for timestamp in timestamps:
        upper_out.append(htr_in[(htr_in.index == timestamp) & (htr_in['inhive_loc'] == 'upper')])
        lower_out.append(htr_in[(htr_in.index == timestamp) & (htr_in['inhive_loc'] == 'lower')])

    # Drop unnecessary columns
    for i in range(len(upper_out)):
        upper_out[i] = upper_out[i].drop(columns=['_measurement'])
        lower_out[i] = lower_out[i].drop(columns=['_measurement'])

    return upper_out, lower_out

def add_transparent_image(background, foreground, x_offset:int=None, y_offset:int=None):
    '''
    Function adapted from https://stackoverflow.com/a/71701023.
    '''
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = alpha_channel[:, :, np.newaxis]

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

    return background


class Hive():
    '''
    This class is meant to hold imaging, thermal and metabolic data for a specific hive at a specific time. It contains the following data:
    - 4 images of the hive (hxr1, hxr2, hxr3, hxr4)
    - 2 ThermalFrames objects (upper, lower)
    - 4 metabolic measures (right, left, upper right, upper left)
    '''

    # Some class variables
    resize_factor = 10.1 # Resize factor for the thermal images relative to the IR images
    inter_htr_dist = 25 # Distance between heaters in pixels
    htr_size=(800,800) # Size of the heaters in pixels (width, height)
    thermal_shifts = [(270,500) if i<2 else (200,505) for i in range(4)]

    def __init__(self, imgs:list, imgs_names:list[str], upper:ThermalFrame, lower:ThermalFrame, metabolic:pd.DataFrame, htr_upper:pd.DataFrame, htr_lower:pd.DataFrame):
        if len(imgs) != 4 or len(imgs_names) != 4:
            raise ValueError("imgs must contain 4 images")
        if metabolic is not None and len(metabolic) != 4:
            raise ValueError("metabolic must contain 4 values")
        
        self.imgs = imgs
        self.imgs_names = imgs_names
        self.upper_tf = upper
        if self.upper_tf is not None:
            self.upper_tf.calculate_thermal_field()
        self.lower_tf = lower
        if self.lower_tf is not None:
            self.lower_tf.calculate_thermal_field()
        self.metabolic = metabolic # pd.DataFrame that has ['ul','ur','ll','lr'] as columns and the metabolic measures as values
        self.htr_upper = htr_upper # pd.DataFrame that has ['status','pwm','avg_temp','obj','actuator_instance'] as columns
        self.htr_lower = htr_lower # pd.DataFrame that has ['status','pwm','avg_temp','obj','actuator_instance'] as columns
        if self.htr_upper is not None and self.htr_lower is not None:
            self.computeHtrPos()    # Computes the heater positions on the RPi images
        self.pp_imgs = None # To store the preprocessed images once computed
        # To store the pixel shifts between the thermal and imaging data. A list of 4 tuples, each tuple containing the x,y shifts for the corresponding RPi image.
        self.co2_pos = {'ul':(300,380),'ur':(4350,380),'ll':(330,380),'lr':(4350,380)}

    def computeHtrPos(self):
        '''
        Computes the positions of the heaters based on self.thermal_shifts.
        affects: 
        - self.htr_pos: dict containing the positions of the heaters for each rpi (top-left and bottom-right corners of the rectangle). The keys are the rpi numbers (0,1,2,3), followed by the heater number (h00 to h09).
        NOTE: positions are for images that have already been flipped horizontally.
        '''
        htr_pos = {}
        for i in range(4):
            htr_pos[i] = {}
            for j in range(10):
                x_pos = self.thermal_shifts[i][0] + self.inter_htr_dist+ (4-j//2) * (self.inter_htr_dist + self.htr_size[0])
                y_pos = self.thermal_shifts[i][1] + self.inter_htr_dist + (j%2) * (self.inter_htr_dist + self.htr_size[1])
                if i < 2:
                    htr_pos[i][f'h{j:02d}'] = ((x_pos, y_pos),(x_pos+self.htr_size[0],y_pos+self.htr_size[1]))
                else:
                    htr_pos[i][f'h{j:02d}'] = (self.imgs[0].shape[1]-x_pos-self.htr_size[0],y_pos),(self.imgs[0].shape[1]-x_pos,y_pos+self.htr_size[1])
        self.htr_pos = htr_pos

    def setCo2Pos(self, co2_pos:dict):
        '''
        Sets the position of the CO2 measures on the images.
        '''
        self.co2_pos = co2_pos

    def setThermalShifts(self, thermal_shifts:list):
        '''
        Sets the pixel shifts between the thermal and imaging data.
        '''
        assert len(thermal_shifts) == 4, "thermal_shifts must contain 4 tuples"
        for shift in thermal_shifts:
            assert len(shift) == 2, "Each tuple in thermal_shifts must contain 2 values"

        self.thermal_shifts = thermal_shifts
        # Re-compute the heater positions
        self.computeHtrPos()

    def _co2_snapshot(self,rgb_imgs:list):
        min_size = 4
        max_size = 10

        for i, img in enumerate(rgb_imgs):
            if i == 0 or i == 2:
                co2_showed = ['ur','ul']
            else:
                co2_showed = ['lr','ll']

            co2_pos = self.co2_pos
            if i >= 2:
                # Flip the co2_pos horizontally
                co2_pos = {k:(img.shape[1]-v[0]-220,v[1]) for k,v in co2_pos.items()}

            for co2 in co2_showed:
                # Compute the size of the text based on the metabolic measure. Put max_size at 30000 and min_size at 300, linearly.
                # First clip the values to be between 360 and 30000
                co2_value = int(np.clip(self.metabolic[co2],360,30000))
                color = (255 * (co2_value - 360) / (30000 - 360),0,0)
                size = min_size + (max_size - min_size) * (co2_value - 360) / (30000 - 360)
                if (co2[1] == 'r' and i<2) or (co2[1] == 'l' and i>=2):
                    # Change the x value of co2_pos to decrease with the size
                    co2_pos[co2] = (co2_pos[co2][0] - int(800*(size-min_size)/(max_size-min_size)), co2_pos[co2][1])
                cv2.putText(rgb_imgs[i], f"{co2_value}", co2_pos[co2], cv2.FONT_HERSHEY_SIMPLEX, size, color, 20, cv2.LINE_AA)

    def _tmp_snapshot(self,rgb_imgs:list, v_min, v_max, thermal_transparency, contours):
        overlays = []
        # Store the max temp coordinates and value
        max_temp = 0
        min_temp = 200
        max_temp_coords = (0,0,0) # First coordinate is the frame, then x and y
        for i,tf in enumerate([self.upper_tf, self.lower_tf]):
            max_temp_tf = np.max(tf.thermal_field)
            min_temp_tf = np.min(tf.thermal_field)
            if min_temp_tf < min_temp:
                min_temp = min_temp_tf
            if max_temp_tf > max_temp:
                max_temp = max_temp_tf
                max_temp_coords = (i,tf.thermal_field.argmax() % tf.thermal_field.shape[1], tf.thermal_field.argmax() // tf.thermal_field.shape[1])
            therm_field_norm = (tf.thermal_field - v_min) / (v_max - v_min)
            # Apply matplotlib colormap (e.g., 'bwr')
            colormap = plt.colormaps['bwr']
            overlay_colored = colormap(therm_field_norm)  # Returns RGBA values in [0, 1]
            # Scale to [0, 255] for OpenCV compatibility
            overlay_rgb = (overlay_colored * 255).astype(np.uint8)
            overlay_rgb[:,:,3] = int(255*thermal_transparency)
            overlay_rgb = cv2.resize(overlay_rgb, (int(overlay_rgb.shape[1] * Hive.resize_factor), int(overlay_rgb.shape[0] * Hive.resize_factor)), interpolation=cv2.INTER_NEAREST)
            overlays.append(overlay_rgb)

        fig, ax = plt.subplots()  # Create a dummy figure to prevent automatic plotting
        _contours = [ax.contour(tf.thermal_field, levels=contours, colors='none') for tf in [self.upper_tf, self.lower_tf]] # Only compute, no color
        plt.close(fig)  # Close the figure to prevent display

        # Extract paths from the contour
        paths = [
            [
                collection.get_paths()[0]
                for collection in contour.collections
                if collection.get_paths()
            ]
            for contour in _contours
        ]

        # Create a blank canvas for OpenCV drawing
        canvas = [np.zeros_like(overlay[:,:,0]) for overlay in overlays]

        tf_shape = self.upper_tf.thermal_field.shape
        # Draw the contours onto the canvas
        for i, path in enumerate(paths): # For each frame
            if path == []:
                continue
            for p in path: # For each path for a given frame
                # Get the vertices and codes
                vertices = p.vertices
                codes = p.codes

                # Initialize a list to store points for each disjoint segment
                segment_points = []

                for vert, code in zip(vertices, codes):
                    if code == Path.MOVETO:
                        # Start of a new segment: draw the previous segment if it exists
                        if segment_points:
                            # Scale and convert to integer pixel coordinates
                            segment_points = np.array(segment_points)
                            segment_points[:, 0] *= canvas[i].shape[1] / tf_shape[1]
                            segment_points[:, 1] *= canvas[i].shape[0] / tf_shape[0]
                            segment_points = np.round(segment_points).astype(np.int32)
                            
                            # Draw the segment as a polyline
                            cv2.polylines(canvas[i], [segment_points], isClosed=False, color=255, thickness=5)
                        
                        # Start a new segment
                        segment_points = [vert]
                    elif code in (Path.LINETO, Path.CLOSEPOLY):
                        # Continue the current segment
                        segment_points.append(vert)
                
                # Draw the last segment if it exists
                if segment_points:
                    segment_points = np.array(segment_points)
                    segment_points[:, 0] *= canvas[i].shape[1] / tf_shape[1]
                    segment_points[:, 1] *= canvas[i].shape[0] / tf_shape[0]
                    segment_points = np.round(segment_points).astype(np.int32)
                    cv2.polylines(canvas[i], [segment_points], isClosed=False, color=255, thickness=5)

        # Overlay the contours onto your RGB overlay
        for i,_ in enumerate(overlays):
            overlays[i][canvas[i] > 0] = (0, 0, 0, 255)  # Black contour with full opacity

        # Put a circular marker at the max temperature coordinates
        cv2.circle(
            overlays[max_temp_coords[0]], 
            (int(max_temp_coords[1] * Hive.resize_factor), int(max_temp_coords[2] * Hive.resize_factor)), 
            20, 
            (255, 0, 0, 255), 
            -1
        )

        # Prepare overlays for each picture
        overlays_flipped = {0:overlays[0],1:overlays[1],2:cv2.flip(overlays[0],1),3:cv2.flip(overlays[1],1)}
        
        for i, bg in enumerate(rgb_imgs):
            overlay_rgb = overlays_flipped[i]
            if i%2 == max_temp_coords[0]:
                coords = (int(max_temp_coords[1] * Hive.resize_factor), int(max_temp_coords[2] * Hive.resize_factor))
                if i>=2:
                    # Flip the coordinates horizontally
                    coords = (overlay_rgb.shape[1]-coords[0],coords[1])
                # Add a 20px margin to the coordinates
                coords = (coords[0]+20,coords[1]-20)
                cv2.putText(overlay_rgb, f"{max_temp:.1f}", coords, cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0, 255), 10, cv2.LINE_AA)
            rgb_imgs[i]= add_transparent_image(bg, overlay_rgb, self.thermal_shifts[i][0], self.thermal_shifts[i][1])
        
        return rgb_imgs, min_temp
    
    def _htr_snapshot(self,rgb_bg:list):
        # Draw a rectangle around the heaters and add information about the heaters
        for i, _ in enumerate(rgb_bg):
            htrs = self.htr_upper if (i == 0 or i == 2) else self.htr_lower
            for htr in [f'h{i:02d}' for i in range(10)]:
                htr_df = htrs[htrs['actuator_instance']==htr]
                pwm = htr_df[htr_df['_field']=='pwm']['_value'].values[0]
                obj = htr_df[htr_df['_field']=='obj']['_value'].values[0]

                # Draw a rectangle around the heater
                color = (255 * pwm / 950,0,0)
                width = int(2 + 7 * pwm / 950)
                mrg = 10 # Just a small padding around the text

                cv2.rectangle(rgb_bg[i], self.htr_pos[i][htr][0], self.htr_pos[i][htr][1], color, width)
                cv2.putText(rgb_bg[i], f"{int(pwm)}           {obj:.1f}", (self.htr_pos[i][htr][0][0]+mrg,self.htr_pos[i][htr][1][1]-mrg), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5, cv2.LINE_AA)
                # Put the heater number on top right of the rectangle
                cv2.putText(rgb_bg[i], htr, (self.htr_pos[i][htr][0][0]+mrg,self.htr_pos[i][htr][0][1]+10*mrg), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 5, cv2.LINE_AA)

    def snapshot(self,thermal_transparency:float=0.25,v_min:float=10,v_max:float=35,contours:list=[]):
        '''
        Generates a global image with the 4 images of the hives with the timestamp on the pictures. It then adds the ThermalFrames ontop of the images.
        '''
        # Preprocess images if not already done
        if self.pp_imgs is None:
            # Preprocess images with Preprocessing library
            self.pp_imgs = []
            for img in self.imgs:
                self.pp_imgs.append(beautify_frame(img))

        rgb_bg = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in self.pp_imgs]
        
        min_temp = -273
        if self.upper_tf is not None and self.lower_tf is not None:
            rgb_bg, min_temp = self._tmp_snapshot(rgb_bg,v_min,v_max,thermal_transparency,contours) # Adds thermal field and isotherms to the images

        if self.htr_upper is not None and self.htr_lower is not None:
            self._htr_snapshot(rgb_bg) # Adds heaters data on the images

        if self.metabolic is not None:
            self._co2_snapshot(rgb_bg) # Add the CO2 measurements on the images

        assembled_img = imageHiveOverview(rgb_bg, self.imgs_names)
        # add ambient temperature on the image (min temp)
        ambient_t_text = f"Ambient: {min_temp:.1f} C"
        (text_width, text_height), _ = cv2.getTextSize(ambient_t_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        rectangle_bgr = (255, 255, 255)
        box_coords = ((2700, 2130 + 15), (2700 + text_width, 2130 - text_height - 15))
        cv2.rectangle(assembled_img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(assembled_img, ambient_t_text, (2700, 2130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

        return assembled_img
    
    def plot_temp(self, frame:str, ax, cmap = None, show_cb:bool = False, contours = None, v_min = None, v_max = None):
        '''
        Plots the thermal data on the images.
        '''
        if frame == "upper":
            return self.upper_tf.plot_thermal_field(ax, cmap, show_cb, contours, v_min, v_max)
        elif frame == "lower":
            return self.lower_tf.plot_thermal_field(ax,cmap, show_cb, contours, v_min, v_max)
        else:
            raise ValueError("frame must be either 'upper' or 'lower'")
        
    def compute_pixel_shifts(self):
        '''
        Computes the pixel shifts between the thermal and the imaging data, for every image of the hive.
        Returns a list of 4 tuples, each tuple containing the x,y shifts for the corresponding RPi image.
        NOTE: This function is currently not used as the line detection method is not reliable.
        '''
        shifts = []
        if self.pp_imgs is None:
            # Preprocess images with Preprocessing library
            self.pp_imgs = []
            for img in self.imgs:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.pp_imgs.append(beautify_frame(gray_img))

        for pp_img in self.pp_imgs:
            # Perform edge detection
            edges = cv2.Canny(pp_img, 350, 450)
            # Detect lines using Hough Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=3500, maxLineGap=1000)
            h_lines = [line for line in lines if abs(line[0][1] - line[0][3]) < 20] # Horizontal lines
            v_lines = [line for line in lines if abs(line[0][0] - line[0][2]) < 20] # Vertical lines
            # Discard the v_lines that are too close to the left border
            if v_lines is not None:
                v_lines = [line for line in v_lines if line[0][0] > 10]

            if len(h_lines) == 0:
                raise ValueError("No horizontal lines detected")
            if len(v_lines) == 0:
                raise ValueError("No vertical lines detected")
            
            lowest_h_line = h_lines[np.argmax([line[0][1] for line in h_lines])]
            leftest_v_line = v_lines[np.argmin([line[0][0] for line in v_lines])]

            upper_shift = np.mean([lowest_h_line[0][1], lowest_h_line[0][3]])
            left_shift = np.mean([leftest_v_line[0][0], leftest_v_line[0][2]])
            shifts.append((left_shift, upper_shift))
        
        self.thermal_shifts = shifts
        return self.thermal_shifts
    
