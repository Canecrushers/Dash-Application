
##
def get_mask_path(tile_x, tile_y):
    path = 'C:\\Users\\kunal\\Desktop\\WORK\\Datathon\\Phase02-DataDelivery\\'
    path += f"masks\\mask-x{tile_x}-y{tile_y}.png"
    #path += f"./data/sentinel-2a-tile-{tile_x}x-{tile_y}y/masks/{mask_type}-mask.png"
    #path = f'E:\work\canecrushers\phase-01\data\sentinel-2a-tile-7680x-10240y\timeseries\'
    return path

	
def get_mask_path(tile_x, tile_y):
    path = 'C:\\Users\\kunal\\Desktop\\WORK\\Datathon\\Phase02-DataDelivery\\'
    path += f"masks\\mask-x{tile_x}-y{tile_y}.png"
#     path = f"./data/sentinel-2a-tile-{tile_x}x-{tile_y}y/masks/{mask_type}-mask.png"
    return path


def load_image(tile_path):
    img = Image.open(tile_path)
    return img


def get_tile_pixels(img):
    pixels = img.load()
    return pixels


def plot_image(img):
    plt.imshow(img)
    

def is_in_mask(mask_pixels, pixel_x, pixel_y):
    if mask_pixels[pixel_y, pixel_x] == (0, 0, 0, 255):
        return True
    else:
        return False
    
def pixels_in_mask(tile_x, tile_y):
    pixel_list = []
    
    mask_path = get_mask_path(tile_x, tile_y)    
    mask_img = load_image(mask_path)    
    mask_pix = get_tile_pixels(mask_img)    
    mask_img_size = mask_img.size
    
    for pixel_x in range(0,mask_img.size[0]):
        for pixel_y in range(0,mask_img.size[1]):
            in_mask = is_in_mask(mask_pix, pixel_x, pixel_y)
            if in_mask:
                pixel_list.append(str(pixel_y)+ " "+ str(pixel_x))
    return pixel_list


	
##Image Functions

def open_image(path, mode = None, cropbox = None, verbose = True):
    if verbose:
        print(path)
    img = Image.open(path)
    if cropbox is not None:
        img = img.crop(cropbox)
    if mode is not None:
        img = img.convert(mode)
    if verbose:
        print("Format: {0}\nSize: {1}\nMode: {2}".format(img.format, img.size, img.mode))
        (width, height) = img.size
        print('width:',width,'height:',height)
    return img
	
# Time series functions

def get_timeseries_image_paths(tile_x, tile_y, band):
    path = 'C:\\Users\\kunal\\Desktop\\WORK\\Datathon\\Phase02-DataDelivery\\'
    path += f"sugarcanetiles\\{tile_x}-{tile_y}-{band}*.png"
    #print(path)
    images = glob.glob(path)
    return images
	
def last_date_in_path(path):
    return re.findall('\d{4}-\d{2}-\d{2}',path)[-1]
	
def write_to_excel(df, file_name, sheet_name='sheet1'):
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    df.to_excel(writer, sheet_name=sheet_name)
    writer.save()
	
def read_img_pixel_values(tile_x, tile_y, date, *args):
    """ return array of arrays 
        one array for each pixel (10m X 10 m), containing 
        an array for each spectrum value for the pixel
    """
    int_max = 2**15-1
    for img in (args):
        assert img.size == args[0].size
    (width, height) = args[0].size
    #print('width',width,'height',height)

    pixl_list = [img.load() for img in args]

    result_list = []
    for x in range(0, width):
        for y in range(0, height):
            val_list = [tile_x, tile_y, x, y, date]
            for pix in pixl_list:
                val = pix[x,y]
                if isinstance(val, tuple):
                    val_list.extend(val)
                else:
                    val_list.append(val)
            #print(val_list)
            result_list.append(val_list)
    return result_list

	
def overlayPredictionImage(df, tci, overlay_colour):
    """ overlay harvest predition onto image
        df - pandas dataframe with 'x', 'y' and 'prediction' integer columns
        tci - rgb image
        overlay_colour - list of np.array([r,g,b], dtype='uint8') representing the colours to overlay
        return a numpy array x by y by [r,g,b]
    """
    result = np.array(tci)
    for row in df.itertuples():
        result[row.y,row.x] = overlay_colour[row.prediction]
    return result