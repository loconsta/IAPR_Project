import PIL.Image
import numpy as np
from typing import Union
from glob import glob
import pandas as pd
import os
#from treys import Card
from termcolor import colored
#from utils import eval_listof_games , debug_listof_games, save_results , load_results

import skimage.io
import matplotlib.pyplot as plt
from skimage.segmentation import flood, flood_fill
from skimage import morphology
from skimage.morphology import closing, opening, disk, square
from skimage import feature
import numpy as np

from skimage import filters
import scipy
import cv2 as cv
import plotly.express as px
import scipy.ndimage as nd

card_titles = ['Kcard', 'Qcard', 'Jcard', '10card', '9card', '8card', '7card', '6card', '5card', '4card', '3card', '2card', 'Acard']
ground_truth_titles = ['K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2', 'A', 'trèfle', 'pique', 'carreau', 'coeur']
    

""""""""""""""""""""""""
"""   Loading stuff  """
""""""""""""""""""""""""

def load_data(paths_list):
    images = []
    for path in paths_list:
        img = skimage.io.imread(path)
        images.append(img)
    return images


def isolate_card_features(cards, kings):
    """ Get all symbols, letters, numbers and original cards """
    # isolate cards
    idx = [520,1190,1190,1190,1190,1840,1840,1840,1890,2530,2530,2550,2550]
    row = 640
    idy = [2750,1940,2460,2980,3480,1915,2450,2950,3500,1930,2470,2975,3500]
    col = 470
    individual_cards = []

    for x,y in zip(idx, idy):
        card = cards[x:x+row, y:y+col]
        individual_cards.append(card)
    # isolate numbers/letters
    idx = np.array([50,50,50,45,50,35,50,50,40,40,45,45,49]) - 3
    row = 70 + 7
    idy = [40,40,35,35,40,50,40,35,40,50,40,40,46]
    col = 50
    numbers = []

    for x,y,card in zip(idx, idy, individual_cards):
        number = card[x:x+row, y:y+col]
        numbers.append(number)
    # isolate symbols
    idx = [1620,1610,2280,2270]
    row = 60
    #idy = [2575,3088,2592,3140]
    idy = [2520,3033,2538,3090]
    col = 45
    symbols = []

    for x,y in zip(idx, idy):
        symbol = kings[x:x+row, y:y+col]
        symbols.append(symbol)

    return individual_cards, numbers, symbols

""""""""""""""""""""""""""""""
"""  Cropping table stuff  """
""""""""""""""""""""""""""""""

def cropping_routine(image):
    """ General function, calls the others """
    x = hist_eq(image)
    x = binarization(x)
    x, cropped_img = crop_table_from_binary(image, x)
    return x, cropped_img


def binarization(image):
    x = skimage.color.rgb2gray(image)
    # smooth for generalization a#nd cleaning
    x = filters.gaussian(x, sigma = 3)
    # binarization
    otsu = filters.threshold_otsu(x)
    x = (x > otsu).astype(int)
    return x

def hist_eq(image):
    channel_1 = skimage.exposure.equalize_hist(image[:,:,0])
    channel_1 = filters.gaussian(channel_1, sigma = 3)
    channel_2 = skimage.exposure.equalize_hist(image[:,:,1])
    channel_2 = filters.gaussian(channel_2, sigma = 3)
    channel_3 = skimage.exposure.equalize_hist(image[:,:,2])
    channel_3 = filters.gaussian(channel_3, sigma = 3)
    output = np.stack([channel_1, channel_2, channel_3], axis = 2)
    return output

def crop_table_from_binary(image, bin_img):
    # basic cleaning of outside white aberrations
    row_med = np.median(bin_img, axis = 0)
    bin_img[:,row_med==0] = 0
    
    # create imperfect mask from contours
    contour = one_contour_by_img([bin_img])[0]
    mask = np.zeros((bin_img.shape))
    mask[contour[:,1], contour[:,0]] = 255
    mask = nd.binary_fill_holes(mask)
        
    # second cleaning of outside defaults (inside is filled now)
    row_med = np.median(mask, axis = 0)
    mask[:,row_med==0] = 0
    col_med = np.median(mask, axis = 1)
    mask[col_med==0,:] = 0
    
    # crop from final correct contour of table
    contour = one_contour_by_img([mask])[0]
    
    # initiate crop dimensions
    left, right = image.shape[1], 0
    up, down = image.shape[0], 0
    # find crop dimensions
    if np.min(contour[:,1]) < up: up = np.min(contour[:,1])
    if np.max(contour[:,1]) > down: down = np.max(contour[:,1])
    if np.min(contour[:,0]) < left: left = np.min(contour[:,0])
    if np.max(contour[:,0]) > right: right = np.max(contour[:,0])
    crop = image[up:down, left:right]
    return mask, crop


""""""""""""""""""""""""""""""
"""     Split table areas     """
""""""""""""""""""""""""""""""
""" GLOBAL CONSTANTS (SCOTCH MARKERS) """
P1 = [0.5, 0.875]
P2 = [0.122, 0.749]
P3 = [0.135, 0.308]
P4 = [0.54, 0.107]
C1 = [0.932, 0.227]
C2 = [0.932, 0.381]
C3 = [0.932, 0.535]
C4 = [0.932, 0.673]
C5 = [0.924, 0.811]
MARKERS = np.array([P1, P2, P3, P4,
                          C1, C2, C3, C4, C5])
CARD_DIM = [0.165, 0.118]

""" Use global constants in functions """

def find_markers_idx(image):
    table_dim = image.shape[:2]
    return (MARKERS * table_dim).astype(int)

def find_player_search_area(image, marker):
    # adapt search area to table size
    row, col = image.shape[:2]
    R, C = int(row/6), int(col/6)
    x1, x2, y1, y2 = marker[0]-R, marker[0]+R, marker[1]-C, marker[1]+C
    # return a crop on the image
    x1, x2 = np.clip([x1, x2], 0, row)
    y1, y2 = np.clip([y1, y2], 0, col)
    search_area = image[x1:x2, y1:y2]
    return search_area

# def find_common_search_area(image, marker):
#     """TO REFINE"""
#     # adapt card size to table size
#     row, col = image.shape[:2]
#     R, C = int((CARD_DIM[0]*row)/2), int((CARD_DIM[1]*col)/2)
#     x1, x2, y1, y2 = marker[0]-R, marker[0]+R, marker[1]-C, marker[1]+C
#     # return a crop on the image
#     x1, x2 = np.clip([x1, x2], 0, row)
#     y1, y2 = np.clip([y1, y2], 0, col)
#     search_area = image[x1:x2, y1:y2]
#     return search_area

def find_common_search_area(image, markers):
    """ Works but a bit edgy, would be better to rely on something else """
    # isolate a 1rst big search area in bottom 3rd of the image
    row, col = image.shape[:2]
    C = int((CARD_DIM[1]*col))
    start, end = markers[0,1]-C, markers[-1,1]+C
    big_area = image[int(2/3*row):row, start:end]
    
    # refine using edge detector and contours filtering
    edges = edge_detector([big_area])[0]
    contours = contours_by_img([edges])[0]
    contours = filter_contours_by_size([contours], 100, 3000)[0]
    start = np.min([np.min(contour[0,:]) for contour in contours])
    end = np.max([np.max(contour[0,:]) for contour in contours])

    # rescale image and divide in 5
    big_area = big_area[:,start:end]
    w = int(big_area.shape[1]/5)
    cards = [big_area[ : , w*i : w*(1+i) ] for i in range(5)]
    return cards

def find_turns_cards(image):
    _, crop = cropping_routine(image)
    #crop = hist_eq(crop[-crop.shape[0]//3:,:])
    sat=skimage.color.rgb2hsv(crop)[:, :, 0]
    edge=edge_detector([sat])[0]
    contours = contours_by_img([edge])[0]

    im = np.zeros((edge.shape[0], edge.shape[1]))
    keep=[]
    for contour in contours:
        if(len(contour)>=600):
            im[contour[:,1], contour[:,0]] = 255
            keep.append(contour)
    contours_poly = [None]*len(keep)
    boundRect = [None]*len(keep)
    for i, c in enumerate(keep):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])#contains index of corners the rectange
        #centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
     # Draw polygonal contour + bonding rects + circles
    drawing = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
    for i in range(len(keep)):
        #cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,255,0), 2)
    cards = []
    for i in range(len(keep)):
        card = crop[int(boundRect[i][1]):int(boundRect[i][1]+boundRect[i][3]),int(boundRect[i][0]):int(boundRect[i][0]+boundRect[i][2])]
        cards.append(card)
    return(cards)
        
    """"""""""""""""""""""""""""""
"""  Contours computation  """
""""""""""""""""""""""""""""""

def one_contour_by_img(img_list):
    """
    Extracts the biggest contour for each image
    return: list(contour_img_1, contour_img_2, ...)
    """
    cont_img = []
    
    #Contours of images
    for img in img_list:
        # Compute all contours in image
        contours, hierarchy = cv.findContours(img.astype(np.uint8) ,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Case 1: 1 contour detected in image
        if len(contours) == 1:
            contour = contours[0][:,0]

        # Case 2: more than 1 contour detected
        else:
            # find and keep biggest contour, as others are artefacts
            big_contour_idx = 0
            for j in range(len(contours)):
                if len(contours[j]) > contours[big_contour_idx].shape[0]:
                    big_contour_idx = j
            contour = contours[big_contour_idx][:,0]

        # Record contour
        cont_img.append(contour)
    return cont_img


def contours_by_img(img_list):
    """ 
    Extracts all contours for each image
    return: list(  list(contour_1_img_1, ..., contour_m_img_1)  ,  ...,  list(contour_1_img_n, ..., contour_m_img_n)  )
    """
    cont_img = []
    
    #Contours of images
    for img in img_list:
        # Compute all contours in image
        contours, hierarchy = cv.findContours(img.astype(np.uint8) ,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        # Case 1: 1 contour detected in image
        if len(contours) == 1:
            contour = contours[0][:,0]
            contour = [contour]
            # append 1 contour in a list
            cont_img.append(contour)
            
            
        # Case 2: more than 1 contour detected
        else:
            # append all contours in a list (necessary reshaping)
            img_contours = [contours[i][:,0] for i in range(len(contours))]
            cont_img.append(img_contours)
        
    return cont_img

  
def filter_contours_by_size(sobel_all_contours, lower_bound, upper_bound):
    all_filtered_contours = []
    for contours in sobel_all_contours:
        image_conts = []
        for contour in contours:
            size = len(contour)
            if size > lower_bound and size < upper_bound:
                image_conts.append(contour)
        all_filtered_contours.append(image_conts)
    return all_filtered_contours


def complex_contours(contour_list):
    contours = []
    for contour in contour_list:
        complex_contour = contour[:,0] + 1j * contour[:,1]
        contours.append(complex_contour)
    return contours

""""""""""""""""""""""""""""""
"""    Features creation   """
""""""""""""""""""""""""""""""

def n_FT_descr(complex_contours, n):
    imgs_fft = []
    for complex_contour in complex_contours:
        fft = np.fft.fft(complex_contour)[:n]
        norm = np.abs(fft)
        imgs_fft.append(norm)
    imgs_fft = np.asarray(imgs_fft, dtype=object)
    return imgs_fft


def edge_detector(color_images):
    final_images = []
    for image in color_images:
        grayscale = skimage.color.rgb2gray(image)
        # smooth for generalization and cleaning
        smoothed = filters.gaussian(grayscale, sigma = 1)
        # edge detector
        edges = filters.sobel(smoothed)
        #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        #edges =cv.morphologyEx(smoothed,cv.MORPH_GRADIENT,kernel)
        
        otsu = filters.threshold_otsu(edges)
        output = edges > otsu
        final_images.append(output)
    return final_images

def get_ground_truth_predictors():
    #First load the ground truth 
    path_data = 'data/image_setup/'
    cards = path_data + 'spades_suits.jpg'
    kings = path_data + 'kings.jpg'
    cards, kings = load_data([cards, kings])
    individual_cards, numbers, symbols = isolate_card_features(cards, kings)
    all_images = individual_cards + numbers + symbols
    #Apply edge detector
    sobel_images = edge_detector(all_images)
    sobel_contours = one_contour_by_img(sobel_images)
    sobel_all_contours = contours_by_img(sobel_images)
    
    #Get upper and lower bound for tolerance
    ground_truth_contours_len = [len(sobel_contours[i]) for i in range(13,30,1)]
    min_size, max_size = np.min(ground_truth_contours_len), np.max(ground_truth_contours_len)
    tol_down, tol_up = 20, 20
    cont_range = [min_size-tol_down, max_size+tol_up]
    lower_bound = cont_range[0]
    upper_bound = cont_range[1]
    
    all_filtered_contours = filter_contours_by_size(sobel_all_contours, lower_bound, upper_bound)
    
    cards_contours = all_filtered_contours[:13]
    ground_truth_contours = [sobel_contours[i] for i in range(13,30,1)]
    n = 10
    # compute ground truth descriptors
    GT_descr = n_FT_descr(complex_contours(ground_truth_contours), n)

    # compute all filtered cards contours descriptors
    cards_comp_ct = [complex_contours(card_cont) for card_cont in cards_contours]
    cards_descr = [n_FT_descr(comp_ct, n) for comp_ct in cards_comp_ct]
    cards_3D_descr = []
    for card_descr in cards_descr:
        descr_1 = card_descr[:,1]
        descr_2 = card_descr[:,2]
        descr_3 = card_descr[:,3]
        descr_4 = card_descr[:,4]
        descr_5 = card_descr[:,5]
        descr_6 = card_descr[:,6]
        descr_7 = card_descr[:,7]
        descr_8 = card_descr[:,8]
        descr_9 = card_descr[:,9]
        card_contours_descr = np.vstack([descr_1,descr_2,descr_3,
                                         descr_4,descr_5,descr_6,
                                        descr_7,descr_8,descr_9]).T
        cards_3D_descr.append(card_contours_descr)

        descr_1 = GT_descr[:,1]
        descr_2 = GT_descr[:,2]
        descr_3 = GT_descr[:,3]
        descr_4 = GT_descr[:,4]
        descr_5 = GT_descr[:,5]
        descr_6 = GT_descr[:,6]
        descr_7 = GT_descr[:,7]
        descr_8 = GT_descr[:,8]
        descr_9 = GT_descr[:,9]

    GT_3D_descr = np.vstack([descr_1,descr_2,descr_3,
                             descr_4,descr_5,descr_6,
                            descr_7,descr_8,descr_9]).T


    return GT_3D_descr, lower_bound, upper_bound
    
    
def get_predictors(cards, lower, upper):
    #bin_cards=[]
    #for card in cards:
    #    bin_cards.append(binarization(card))
    sobel_test_images = edge_detector(cards)
    test_contours = one_contour_by_img(sobel_test_images)
    test_all_contours = contours_by_img(sobel_test_images)
    test_filtered_contours = filter_contours_by_size(test_all_contours, lower, upper)
    cards_comp_ct = [complex_contours(card_cont) for card_cont in test_filtered_contours]
    test_descr = [n_FT_descr(comp_ct, 10) for comp_ct in cards_comp_ct]
    test_3D_descr = []
    for card_descr in test_descr:
        descr_1 = card_descr[:,1]
        descr_2 = card_descr[:,2]
        descr_3 = card_descr[:,3]
        descr_4 = card_descr[:,4]
        descr_5 = card_descr[:,5]
        descr_6 = card_descr[:,6]
        descr_7 = card_descr[:,7]
        descr_8 = card_descr[:,8]
        descr_9 = card_descr[:,9]
        card_contours_descr = np.vstack([descr_1,descr_2,descr_3,
                                         descr_4,descr_5,descr_6,
                                        descr_7,descr_8,descr_9]).T
        test_3D_descr.append(card_contours_descr)
    return test_3D_descr
""""""""""""""""""""""""""""""
"""      Predictions       """
""""""""""""""""""""""""""""""

def predict_cards_from_predictors(cards_3D_descr):
    #Need to find a way to differentiate 6 and 9
    #Get GT predictors and dictionary
    GT_3D_descr = get_ground_truth_predictors()[0]
    number_key = {0 : 'K', 1 : 'Q', 2 : 'J', 3 : '10', 4 : '9',
            5 : '8', 6 : '7', 7 : '6', 8 : '5', 9 : '4',
            10 : '3', 11 : '2', 12 : 'A'}
    symbol_key = {0 : 'trèfle', 1 : 'pique', 2 : 'carreau', 3 : 'coeur'}
    
    pred_numbers = []
    pred_symbols = []
    for card_all_descr in cards_3D_descr:
        # for each number, compute minimal distance to it
        dist_to_numbers = []
        for i in range(GT_3D_descr.shape[0]-4):
            diff = card_all_descr - GT_3D_descr[i,:]
            dist = np.linalg.norm(diff.astype(np.float), axis = 1)
            dist_to_numbers.append(np.min(dist))
        idx = np.argmin(dist_to_numbers)
        #if (idx==4 or idx==7):
            #diff_6_9()
        pred_numbers.append(number_key[idx])
        
        # for each symbol, compute minimal distance to it
        dist_to_symbols = []
        for i in range(GT_3D_descr.shape[0]-4, GT_3D_descr.shape[0], 1):
            diff = card_all_descr - GT_3D_descr[i,:]
            #dist = diff[:,0]**2 + diff[:,1]**2 + diff[:,2]**2
            dist = np.linalg.norm(diff.astype(np.float), axis = 1) # more general
            dist_to_symbols.append(np.min(dist))
        idx = np.argmin(dist_to_symbols)
        pred_symbols.append(symbol_key[idx])
        
    return pred_numbers, pred_symbols

def predict_turns(image):
    #_, table = cropping_routine(image)
    #markers = find_markers_idx(table)
    #common_areas = find_common_search_area(table, markers[4:])
    #cards = []
    #for area in common_areas:
    #    cards.append(area)
    cards = find_turns_cards(image)
    GT_3D_descr, lower, upper = get_ground_truth_predictors()
    predictors = get_predictors(cards, lower, upper)
    pred_numbers, pred_symbols = predict_cards_from_predictors(predictors)
    keys=["T1", "T2", "T3", "T4", "T5"]
    turns = dict(zip(keys, zip(pred_numbers,pred_symbols)))

    return turns
    
    
""""""""""""""""""""""""""""""
"""     Plotting stuff     """
""""""""""""""""""""""""""""""

def plot_coutours_length_distrib(cards_contours_len, ground_truth_contours_len):
    # plot ground truth distrib
    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(ground_truth_contours_len, bins=len(ground_truth_contours_len))
    plt.text(400,3.5, f'min = {np.min(ground_truth_contours_len)} \nmax = {np.max(ground_truth_contours_len)}')
    for i, (x, card) in enumerate(zip(ground_truth_contours_len, ground_truth_titles)):
        plt.text(x, (i+1)*0.1, card)
    plt.title('Distribution of ground truth contour lengths')
    plt.show()


    # plot cards distrib
    fig, axes = plt.subplots(ncols = 3, nrows = 5, figsize=(20,20))
    for ax, contours, title in zip(axes.flatten(), cards_contours_len, card_titles):
        lengths = [len(ct) for ct in contours]
        ax.hist(lengths, bins=len(lengths))
        ax.set_title(title)
    axes[-1,-1].axis('off')
    axes[-1,-2].axis('off')
    fig.suptitle('Distribution of contour length on cards')
    plt.show()

    
def plot_fourier_descr_and_card_contours(GT_descr, cards_descr = None, card_idx = None):
    fig, axes = plt.subplots(ncols = 2, figsize=(12,5))

    # CHOOSE WHICH CARD CONTOUR TO PLOT ON FOURIER SPACE
    if card_idx is not None and cards_descr is not None:
        j = card_idx
        axes[0].scatter(cards_descr[j][:,0], cards_descr[j][:,1],
                        color = 'k', label = card_titles[j], alpha = 0.3)
        axes[1].scatter(cards_descr[j][:,0], cards_descr[j][:,1],
                        color = 'k', label = card_titles[j], alpha = 0.3)

    # PLOT GROUND TRUTH FOURIER COEFS
    for descr, label in zip(GT_descr[:-4], ground_truth_titles[:-4]):
        axes[0].scatter(descr[0], descr[1], label = label)

    axes[0].legend(bbox_to_anchor=(1,1))
    axes[0].set_xlabel('1st descriptor')
    axes[0].set_ylabel('2nd descriptor')
    axes[0].set_title('Numbers/Letters Fourier descriptors')

    for descr, label in zip(GT_descr[-4:], ground_truth_titles[-4:]):
        axes[1].scatter(descr[0], descr[1], label = label)
    axes[1].legend(bbox_to_anchor=(1,1))
    axes[1].set_xlabel('1st descriptor')
    #axes[1].set_ylabel('2nd descriptor')
    axes[1].set_title('Symbols Fourier descriptors')
    plt.show()
    
def plot_fourier_descr_3D(GT_descr): 
    ax = plt.axes(projection='3d')
    for descr, label in zip(GT_descr[:-4], ground_truth_titles[:-4]):
        ax.scatter3D(descr[0], descr[1], descr[2], label = label)
    ax.set_title('Letters/Numbers 3D Fourier space')
    ax.set_xlabel('1st descriptor')
    ax.set_ylabel('2nd descriptor')
    ax.set_zlabel('3rd descriptor')
    ax.legend(bbox_to_anchor=(2,1))
    plt.show()

    ax = plt.axes(projection='3d')
    for descr, label in zip(GT_descr[-4:], ground_truth_titles[-4:]):
        ax.scatter3D(descr[0], descr[1], descr[2], label = label)
    ax.set_title('Symbols 3D Fourier space')
    ax.set_xlabel('1st descriptor')
    ax.set_ylabel('2nd descriptor')
    ax.set_zlabel('3rd descriptor')
    ax.legend(bbox_to_anchor=(2,1))
    plt.show()
    
def plot_interactive_3D_descr(df):
    fig = px.scatter_3d(df[:-4], x='descr 1', y='descr 2', z='descr 3')
    #fig.write_html('first_figure.html', auto_open=False)
    fig.show()
    fig = px.scatter_3d(df[-4:], x='descr 1', y='descr 2', z='descr 3')
    #fig.write_html('first_figure.html', auto_open=False)
    fig.show()

# def plot_card(idx, idy, r, c):
#     fig, axes = plt.subplots(ncols=4, figsize=(10,8))
#     for ax,x,y in zip(axes.flatten(),idx, idy):
#         ax.imshow(cards[x:x+r,y:y+c])
#         ax.axis('off')
#     plt.show()


    
""""""""""""""""""""""""""""""
""" Common cards detection """
""""""""""""""""""""""""""""""

def find_cards(crop, sig):
    #crop = cropping_routine(image)[1]#get table area
    size_lim = 2*(CARD_DIM[0]*crop.shape[0]+CARD_DIM[1]*crop.shape[1])-350#check that contour is bigger than the shape of a card (-350 for tolerance)
    value = cv.cvtColor(crop, cv.COLOR_RGB2HSV)[:, :, 2]#keep the value in the hsv channels
    # Compute the Canny filter
    edges2 = feature.canny(value, sigma=sig)

    # Convert the boolean image into a binary (0,1)
    edges2_binary = np.zeros_like(value)
    #edges2_binary = cv.morphologyEx(edges2_binary, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(1,250)))
    edges2_binary[edges2>0] = 1
    
    #Compute contours
    contours = contours_by_img([edges2_binary])[0]
    #Find which contours to keep
    keep=[]
    for contour in contours:
        if(len(contour)>=size_lim):
            #print(len(contour))
            keep.append(contour)
    #Create empty lists to contain contours and angles position
    contours_poly = [None]*len(keep)
    boundRect = [None]*len(keep)
    for i, c in enumerate(keep):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)#Approximate form
        boundRect[i] = cv.boundingRect(contours_poly[i])#contains index of corners the rectange
    boundRect.sort()#sort by top left angle
    cards = []
    for i in range(len(keep)):
        if(380<boundRect[i][2]<480 & 350<boundRect[i][3]):#Check that the bounding rectangle is approximately as large and at least half as high as a card
            #Crop the cards from the original bottom third of image
            print()
            card = crop[int(boundRect[i][1]):int(boundRect[i][1]+boundRect[i][3]),int(boundRect[i][0]):int(boundRect[i][0]+boundRect[i][2])]
            cards.append(card)
    return cards

def find_5_cards(image):
    crop = image[-image.shape[0]//3:,:]#get bottom third of image
    final_cards = []
    best_idx = (0,0)
    best_len = 0
    for sig in range (2,5):
        cards = find_cards(crop,sig)
        #If we find 5 cards, return results
        if len(cards)==5 :
            return cards
    
    #If we don't have 5 cards, try loris' method
    if(best_len!=5):
        new_img = isolate_common_cards(crop)
        for sig in range (2,5):
            cards = find_cards(new_img, sig)
            #If we find 5 cards, record sig and exit
            if len(cards)==5 :
                return cards
    #If both methods don't return good results, return basic separation (/5)
    return find_common_search_area(image, find_markers_idx(image)[4:])

def isolate_common_cards(img):
    x = np.copy(img)
    cond = x[:,:,1] < 222
    x[cond] = 0
    x = skimage.color.rgb2gray(x)
    k = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    x = skimage.morphology.binary_opening(x, k)#, footprint=None, out=None)
    k = cv.getStructuringElement(cv.MORPH_CROSS,(6,6))
    x = skimage.morphology.binary_closing(x, k)#, footprint=None, out=None)
    im = np.zeros((x.shape[0], x.shape[1]))
    
    [cts] = contours_by_img([x])
    lengths = [len(ct) for ct in cts]
    idx = np.argsort(lengths)
    cts_of_cards = ([cts[i] for i in idx])[-5:]
    for ct in cts_of_cards:
        im[ct[:,1], ct[:,0]] = 255
    im = nd.binary_fill_holes(im)
    #plt.imshow(im, cmap = 'gray')
    #plt.show()
    return img*im[...,np.newaxis]