# This is a program to identify the max occupied color and its name from the given image.

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from colory.color import Color
from skimage.color import rgb2lab, deltaE_cie76
import os


# Function to convert the RGB values into Hexa-Decimal values
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

# Function to read the Image in OpenCV and convert its color from BGR to RGB
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    x = 8
    y = 8
    w = 122
    h = 58
    #image = image[y:y+h, x:x+w]
    #cv2..imshow("get_image",image)
    #plt.imshow(image)
    #plt.show()
    return image


# Function to convert the centered HSV cluster into RGB values
def convert_Lab2RGB(center_colors,rgbarr=[]):
    #print("Inside HSV2RGB - center colors =======",center_colors)
    #print("shape = ",center_colors.shape)
    for cluster in center_colors:
        #print("cluster %%%%%%%%%%",cluster)
        l = cluster[0]
        a = cluster[1]
        b = cluster[2]
        #print("H,S,V ========",h,s,v)
        # hsv2rgb = colorsys.hsv_to_rgb(h, s, v) 
        lab_image = np.uint8([[[l,a,b ]]]) 
        #print("hsv_image ***************",hsv_image)
        lab2rgb = cv2.cvtColor(lab_image,cv2.COLOR_Lab2RGB)

        l1 = lab2rgb[0][0][0]
        a1 = lab2rgb[0][0][1]
        b1 = lab2rgb[0][0][2]
        #print("h1,s1,v1&&&&&&&&&&&&&&&&&&&&&&&&&",h1,s1,v1)
        rgb_row = [l1,a1,b1]
        #print("RGBrow =============",rgbrow)
        rgbarr.append(rgb_row)
        #print("lab2rgb arr *******************",rgbarr)
    
    return rgbarr

def get_closest_color_name(dominant):
    #print("Dominant ==",dominant)
    # Red Color
    red =  np.uint8([[[150, 90, 100]]])
    # Blue Color
    blue = np.uint8([[[0,0,128]]])
    white = np.uint8([[[200,255,255]]])
    black = np.uint8([[[0,0,0]]])
    gray = np.uint8([[[128,128,128]]])
    
    lime = np.uint8([[[0,255,0]]])
    yellow = np.uint8([[[255,255,0]]])
    cyan_aqua = np.uint8([[[0,255,255]]])
    magenta = np.uint8([[[255,0,255]]])
    silver = np.uint8([[[192,192,192]]])
    
    maroon = np.uint8([[[128,0,0]]])
    olive = np.uint8([[[128,128,0]]])
    green = np.uint8([[[0,128,0]]])
    purple = np.uint8([[[128,0,128]]])
    teal = np.uint8([[[0,128,128]]])
    navy = np.uint8([[[0,0,128]]])
    

    # Convert from RGB to Lab Color Space
    color_car_rgb =np.uint8([[[dominant[0], dominant[1], dominant[2]]]]) 
    #print("Color_car _RGB ===", color_car_rgb)

    color_red_lab = rgb2lab(red)
    color_blue_lab = rgb2lab(blue)
    color_black_lab = rgb2lab(black)
    color_white_lab = rgb2lab(white)
    color_gray_lab = rgb2lab(gray)
    
    color_lime_lab = rgb2lab(lime)
    color_cyan_aqua_lab = rgb2lab(cyan_aqua)
    color_magenta_lab = rgb2lab(magenta)
    color_yellow_lab = rgb2lab(yellow)
    
    color_silver_lab = rgb2lab(silver)
    color_maroon_lab = rgb2lab(maroon)
    color_olive_lab = rgb2lab(olive)
    color_green_lab = rgb2lab(green)
    color_purple_lab = rgb2lab(purple)
    color_teal_lab = rgb2lab(teal)
    color_navy_lab = rgb2lab(navy)
    
    color_car_lab = rgb2lab(color_car_rgb)

    # Convert from RGB to Lab Color Space
    delta_e_red = deltaE_cie76(color_red_lab, color_car_lab)
    delta_e_blue = deltaE_cie76(color_blue_lab, color_car_lab)
    delta_e_black = deltaE_cie76(color_black_lab, color_car_lab)
    delta_e_white = deltaE_cie76(color_white_lab, color_car_lab)
    delta_e_gray = deltaE_cie76(color_gray_lab, color_car_lab)
    
    delta_e_lime = deltaE_cie76(color_lime_lab , color_car_lab)
    delta_e_cyan_aqua = deltaE_cie76(color_cyan_aqua_lab , color_car_lab)
    delta_e_magenta = deltaE_cie76(color_magenta_lab, color_car_lab)
    delta_e_yellow = deltaE_cie76(color_yellow_lab, color_car_lab)
    
    delta_e_silver = deltaE_cie76(color_silver_lab, color_car_lab)
    delta_e_maroon = deltaE_cie76(color_maroon_lab, color_car_lab)
    delta_e_olive = deltaE_cie76(color_olive_lab, color_car_lab)
    delta_e_green = deltaE_cie76(color_green_lab, color_car_lab)
    delta_e_purple = deltaE_cie76(color_purple_lab, color_car_lab)
    delta_e_teal = deltaE_cie76(color_teal_lab, color_car_lab)
    delta_e_navy = deltaE_cie76(color_navy_lab, color_car_lab)

    # print("-------red", delta_e_red, "blue", delta_e_blue, "white", delta_e_white, "black", delta_e_black)
    

    dicti = {"red": delta_e_red, "blue": delta_e_blue, "white": delta_e_white, "black": delta_e_black, "White": delta_e_white, "Gray": delta_e_gray, "Lime": delta_e_lime, "Cyan/Aqua": delta_e_cyan_aqua, "Magenta": delta_e_magenta, "Yellow": delta_e_yellow, "Silver": delta_e_silver, "Maroon": delta_e_maroon, "Olive": delta_e_olive, "Green": delta_e_green, "Purple": delta_e_purple, "Teal": delta_e_teal, "Navy": delta_e_navy}
    key_min = min(dicti, key=dicti.get)
    #print("Closest Color = ", key_min)
    return key_min


# Function to find the Most Occupied color using K-Means Algorithm .
def get_colors(image, number_of_colors, show_chart):
    #cv2.imshow("Actual image ",image)
    # Resize and Reshape the image
    modified_image = cv2.resize(image, (100, 100), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    # apply K-means algorithm - pass the no. of max. colors to find as argument .
    clf = KMeans(n_clusters = number_of_colors)
    # Fit and predict the labels for the mentioned no. of clusters. Compute cluster centers and predict cluster index for each sample.
    labels = clf.fit_predict(modified_image)
    
    #print("Labels =====",labels)
    # Use Counter to store the no. of pixels available in each cluster
    # A Counter is a subclass of dict. Therefore it is an unordered collection where elements and their respective count are stored as dictionary. 
    counts = Counter(labels)
    #print("counts before sort = = = ",counts)
    
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    #print("counts after sort = = = ",counts)
    #  finding the center color value for each of the clusters
    center_colors = clf.cluster_centers_ 
    #print("centerd_colors =========",center_colors)
    
    #convert the cluster values from HSV2RGB 
    center_colors = convert_Lab2RGB(center_colors,[])
    #print("centerd_colors =========",center_colors)
    
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    #for i in counts.items():
        #print("hex_col list",i)
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()] # This returns the Hexa-Decimal value of each cluster
    #hex_colors = [get_colour_name(ordered_colors[i]) for i in counts.keys()] # This returns the color name as value of each cluster
    
    #print("hex_colors            ",hex_colors)
    Color_names = [Color(i,'xkcd').name for i in hex_colors]
    #print("Color names === ",Color_names)

    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    #print(" rgb_colors : ", rgb_colors)      
    
    
    # TO Display the highly occupied color name 
    global max_occupied_color
    for i,val in Counter(labels).most_common(1):
        max_color = i
        #print("i = ",max_color)
        #print("val = ",val)
        #max_occupied_color = get_closest_color_name(ordered_colors[max_color])
        #max_occupied_color = get_colour_name(ordered_colors[max_color])
        
        #color_name = get_colour_name(ordered_colors[max_color])
        color_name = Color(hex_colors[max_color],'xkcd')
        
        print("Exact color name =",color_name.name)
        print("Hexa value =",hex_colors[max_color])

        max_occupied_color = color_name.name
        
    # To Display each cluster color in pie chart with percentage
    if(show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = Color_names, colors = hex_colors,autopct='%1.0f%%')
        plt.show()
    
    return rgb_colors

# Function to label the identified color name on image
def get_color_label(image):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (20, 20) 
    # fontScale 
    fontScale = 0.5
   
    # Blue color in BGR 
    color = (255, 0, 0) 
  
    # Line thickness of 2 px 
    thickness = 1
    
    # Using cv2.putText() method 
    #image = cv2.putText(image,max_occupied_color, org, font,fontScale, color, thickness, cv2.LINE_AA) 
    image = cv2.putText(cv2.cvtColor(image, cv2.COLOR_Lab2RGB),max_occupied_color, org, font,fontScale, color, thickness, cv2.LINE_AA,bottomLeftOrigin=False) 
    plt.imshow(image)
    plt.show()


IMAGE_DIRECTORY = 'images\car'
#dir_path = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
#print("dir_path =========",dir_path)
Image_Folder_Path =os.path.join(dir_path,IMAGE_DIRECTORY)
#print(Image_Folder_Path)



for file in os.listdir(Image_Folder_Path):
    
    if not file.startswith('.') and file.endswith('.jpg'):
        #print("PAth = ",os.path.join(Image_Folder_Path, file))
        print("PAth = ", file)
        Image_Path = os.path.join(Image_Folder_Path, file)
        #crop_img = img[y:y+h, x:x+w]
        image1 = get_image(Image_Path)
        get_colors(image1,8,True)
        get_color_label(image1)