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
from skimage import feature
from skimage.morphology import closing, opening, disk, square
import numpy as np

from skimage import filters
import scipy
import cv2 as cv
import plotly.express as px
import scipy.ndimage as nd
from skimage.filters import threshold_multiotsu

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
"""  Cropping areas stuff  """
""""""""""""""""""""""""""""""

def find_markers_idx(image, MARKERS):
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
 
def check_if_back(edges, LOWER_BOUND, UPPER_BOUND):
    ct_number = []
    for img in edges:
        k = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        x = skimage.morphology.dilation(img, k)
        [contours] = contours_by_img([x])
        [filtered_contours] = filter_contours_by_size([contours], LOWER_BOUND, UPPER_BOUND)
        ct_number.append(len(filtered_contours))
    is_back_of_card = np.asarray(ct_number) < 4
    return is_back_of_card

def find_common_search_area_v1(image, markers, CARD_DIM):
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


def contours_by_img(img_list, mode = cv.RETR_EXTERNAL):
    """ 
    Extracts all contours for each image
    return: list(  list(contour_1_img_1, ..., contour_m_img_1)  ,  ...,  list(contour_1_img_n, ..., contour_m_img_n)  )
    """
    cont_img = []
    
    #Contours of images
    for img in img_list:
        # Compute all contours in image
        contours, hierarchy = cv.findContours(img.astype(np.uint8) , mode, cv.CHAIN_APPROX_NONE)
        
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


def check_contours(contours, edges):
    final_contours = []
    for contour, edge in zip(contours, edges):
        if len(contour) == 1:
            #print('A')
            #print(contour)
            [alt_contour] = contours_by_img([edge], mode = cv.RETR_LIST)
            #print(alt_contour)
            final_contours.append(alt_contour)
        else: final_contours.append(contour)
    return final_contours

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
        if len(image.shape) == 3: x = skimage.color.rgb2gray(image)
        else: x = image
        # smooth for generalization and cleaning
        x = filters.gaussian(x, sigma = 1)
        # edge detector
        x = filters.sobel(x)
        #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        #edges =cv.morphologyEx(smoothed,cv.MORPH_GRADIENT,kernel)
        
        otsu = filters.threshold_otsu(x)
        output = x > otsu
        final_images.append(output)
    return final_images
    
""""""""""""""""""""""""""""""
"""      Predictions       """
""""""""""""""""""""""""""""""

def predict_cards_from_predictors(descr, contours, GT_descr, number_keys, symbol_keys, K=1):
    pair, location, cont = identify_K_pairs(descr, contours, GT_descr, number_keys, symbol_keys, K)
    card_ID = six_or_nine_check(pair, cont, location)
    
    # check that it returns something (avoid breaking everything)
    if len(card_ID) == 0: card_ID = ['0']
    
    return card_ID


def identify_K_pairs(descr, contours, GT_descr, number_keys, symbol_keys, K):
    """ Create K (here 1 or 2) pairs of number/symbol with their associated contours and location """
    # separate number and symbols descr
    NB_descr = GT_descr[:-4]
    SYM_descr = GT_descr[-4:]
    
    """ Attribute 1 number to each contour """
    # for each contour, compute distance of every number descr to it
    ct_to_nb_dist = []
    nb_cont = []
    nb_keys = []
    # iterate over each contour
    for ct_descr, ct in zip(descr, contours):
        diff = ct_descr - NB_descr
        dist = np.linalg.norm(diff.astype(float), axis = 1)
        # pick most likely number fo contour and record
        idx = np.argmin(dist)
        ct_to_nb_dist.append(dist[idx])
        nb_cont.append(ct)
        nb_keys.append(number_keys[idx])

    """ Attribute 1 symbol to each contour """
    # for each contour, compute distance of every symbol descr to it
    ct_to_sym_dist = []
    sym_cont = []
    sym_keys = []
    # iterate over each contour
    for ct_descr, ct in zip(descr, contours):
        diff = ct_descr - SYM_descr
        dist = np.linalg.norm(diff.astype(float), axis = 1)
        # pick most likely symbol fo contour and record
        idx = np.argmin(dist)
        ct_to_sym_dist.append(dist[idx])
        sym_cont.append(ct)
        sym_keys.append(symbol_keys[idx])

    """ Define true K symbol(s), EASIER TO ISOLATE than numbers (need 3) """
    # compute center of contours to approximate location
    ct_locs = np.array([np.mean(contour, axis = 0) for contour in contours])

    sorted_idx = np.argsort(ct_to_sym_dist)
    sym_cont = ([sym_cont[i] for i in sorted_idx])[:K]
    sym_keys = ([sym_keys[i] for i in sorted_idx])[:K]
    sym_locs = ([ct_locs[i] for i in sorted_idx])[:K]

    """ take min distance number-symbol pairs to find the K associated number of interest """
    candidate_pairs = []
    candidate_locations = []
    candidate_nb_cont = []
    for sym_ct, sym_loc, sym_key in zip(sym_cont, sym_locs, sym_keys):
        # for each candidate symbol, compute distance to candidate number
        dist = [scipy.spatial.distance.cdist(sym_ct, nb_ct).min() for nb_ct in nb_cont]
        # sort distances with number keys and means accordingly
        idx = np.argsort(dist)
        sorted_nb_key = [nb_keys[i] for i in idx]
        sorted_nb_cont = [nb_cont[i] for i in idx]
        sorted_nb_locs = ct_locs[idx] 
        
        # create minimal distance pair, record location for later and
        # make sure we dont create a pair of similar contours by taking [1]
        # and keep number contour for a 6or9 check if needed
        candidate_pairs.append(sorted_nb_key[1]+sym_key)
        candidate_locations.append((sorted_nb_locs[1] + sym_loc)/2)
        candidate_nb_cont.append(sorted_nb_cont[1])

    return candidate_pairs, candidate_locations, candidate_nb_cont


def player_pred(descr, contours, GT_descr, player_id, number_keys, symbol_keys, K=3):
    """ Gets the 3 potential number/symbol pairs and keeps 2 with their location """
    
    # check back of cards case
    cards = ['0','0']
    if not (descr == np.zeros(9)).all():
        candidate_pairs, candidate_locations, candidate_nb_cont = identify_K_pairs(descr, contours,
                                                                                   GT_descr, number_keys,
                                                                                   symbol_keys, K)

        # take 2 different pairs amond the 3
        if candidate_pairs[0] != candidate_pairs[1]:
            cards_ID = [candidate_pairs[0], candidate_pairs[1]]
            locations = np.array([candidate_locations[0], candidate_locations[1]])
            nb_contours = [candidate_nb_cont[0], candidate_nb_cont[1]]
        else:
            cards_ID = [candidate_pairs[0], candidate_pairs[2]]
            locations = np.array([candidate_locations[0], candidate_locations[2]])
            nb_contours = [candidate_nb_cont[0], candidate_nb_cont[2]]

        """ check if number is a 6 or 9 if case occurs """
        cards_ID = six_or_nine_check(cards_ID, nb_contours, locations)

        """ identify which card is where """ #using locations and player ID
        if player_id in [1,4]:
            up_idx = np.argmin(locations[:,1])
            down_idx = np.argmax(locations[:,1])
            up_card = cards_ID[up_idx]
            down_card = cards_ID[down_idx]
            cards = [down_card, up_card]

        if player_id in [2,3]:
            left_idx = np.argmin(locations[:,0])
            right_idx = np.argmax(locations[:,0])
            left_card = cards_ID[left_idx]
            right_card = cards_ID[right_idx]
            cards = [right_card, left_card]
    
    # check that it returns something (avoid breaking everything)
    for card in cards:
        if len(card) == 0: card = ['0']
        
    return cards
    
def six_or_nine_check(cards_ID, nb_contours, locations):
    """ Choose 6 or 9 by looking at distance between (rot) number and mean sym/nb location """
    output_ID = []
    for ID, ct, loc in zip(cards_ID, nb_contours, locations):
        final_ID = ID
        if ID[0] == '6' or ID[0] == '9':
            # rotation of 180° around centr of mass (remains identical)
            rot_ct = rotate_contour(ct)
            # compute center of contour and rot contour (changes)
            Cx = (np.max(ct[:,0]) + np.min(ct[:,0])) // 2
            Cy = (np.max(ct[:,1]) + np.min(ct[:,1])) // 2
            rot_Cx = (np.max(rot_ct[:,0]) + np.min(rot_ct[:,0])) // 2
            rot_Cy = (np.max(rot_ct[:,1]) + np.min(rot_ct[:,1])) // 2
            
            # compute distance
            dist = np.linalg.norm([Cx, Cy] - loc)
            rot_dist = np.linalg.norm([rot_Cx, rot_Cy] - loc)
            if dist > rot_dist: final_ID = '9' + ID[1]
            else: final_ID = '6' + ID[1]
        output_ID.append(final_ID)
    return output_ID
            
def rotate_contour(contour, angle = np.pi):
    Ox, Oy = np.mean(contour, axis = 0)
    rot_x = np.asarray([Ox + np.cos(angle) * (x - Ox) - np.sin(angle) * (y - Oy) for (x,y) in contour], dtype =object)
    rot_y = np.asarray([Oy + np.sin(angle) * (x - Ox) + np.cos(angle) * (y - Oy) for (x,y) in contour], dtype =object)
    rot_contour = np.vstack([rot_x, rot_y]).T
    return rot_contour
    
    
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
"""     Chips stuff     """
""""""""""""""""""""""""""""""
CHIPS_AREA = 50000/(1750*1760)
luce = 221.26
noluce = 86.66

def find_chips_search_area(image):
    # adapt search area to table size
    row, col = image.shape[:2]
    R, C = int(row/4), int(col/4)
    R_2 = 3*R
    C_2 = 3*C
    # return a crop on the image
    search_area = image[R:R_2, C:C_2]
    return search_area


def round_(x,T): ## round function with threshold
    
    n = np.floor(x)
    if (x-n > T): n += 1
    
    return int(n)
    
def n_chips(N,chips_area): ## number of chips
    x = chips_area.shape[0]
    y = chips_area.shape[1]
    
    
    x = N/(x*y)/CHIPS_AREA
    
    return round_(x,T=0.3)

def give_color(chips_area,final_mask): ## give color to each pixel
    
    chips_area_hsv = cv.cvtColor(chips_area, cv.COLOR_RGB2HSV)
    result = np.zeros(chips_area_hsv[:,:,0].shape)
    
    
    chips_area_hsv = np.float64(chips_area_hsv)
    
    x = np.mean(chips_area[0:10,0:10,2])
    black = int(72/(luce-noluce) * (x-noluce) + 38)

    result[chips_area_hsv[:,:,2] < black] = 1 #black
    chips_area_hsv[result!=0,:] = float("NAN")
    
    result[chips_area_hsv[:,:,1] < 60] = 2 #white
    chips_area_hsv[result!=0,:] = float("NAN")
    
    result[(chips_area_hsv[:,:,0] < 10) + (chips_area_hsv[:,:,0] >160)] = 3 #red
    chips_area_hsv[result!=0,:] = float("NAN")
    
    result[(chips_area_hsv[:,:,0] > 95) * (chips_area_hsv[:,:,0] < 130)] = 4 #blue
    chips_area_hsv[result!=0,:] = float("NAN")
    
    result[(chips_area_hsv[:,:,0] > 35) * (chips_area_hsv[:,:,0] < 95)] = 5 #green
    
    result = result*final_mask
    
    return result

def predict_chips_area(chips_area): ## predict the pixels in a chips
    
    chips_area_hsv = cv.cvtColor(chips_area, cv.COLOR_RGB2HSV)
    thresholds = threshold_multiotsu(chips_area_hsv[:,:,2], classes = 2)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(chips_area_hsv[:,:,2], bins=thresholds)

    thresholds1 = threshold_multiotsu(chips_area_hsv[:,:,0], classes = 2)

    # Using the threshold values, we generate the three regions.
    regions1 = np.digitize(chips_area_hsv[:,:,0], bins=thresholds1)

    tot = (1-regions) + regions1

    tot[tot>0.5] = 1
    
    nb_px = np.sum(tot == 1)
    
    x = chips_area.shape[0]
    y = chips_area.shape[1]
    
    
    x = nb_px/(x*y)/CHIPS_AREA
    
    return round_(x,T=0.2),tot

""""""""""""""""""""""""""""""
""" Common cards detection """
""""""""""""""""""""""""""""""

def find_cards(crop, sig, CARD_DIM):
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
            keep.append(contour)
            
    #Create empty lists to contain contours and angles position
    contours_poly = [None]*len(keep)
    boundRect = [None]*len(keep)
    for i, c in enumerate(keep):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)#Approximate form
        boundRect[i] = cv.boundingRect(contours_poly[i])#contains index of corners the rectange
    boundRect.sort()#sort by top left angle
    
    #Empty list for results
    cards = []
    for i in range(len(keep)):
        if(380<boundRect[i][2]<480 and 350<boundRect[i][3]<800):#Check that the bounding rectangle is approximately as large and at least half as high as a card
            #Crop the cards from the original bottom third of image)
            card = crop[int(boundRect[i][1]):int(boundRect[i][1]+boundRect[i][3]),int(boundRect[i][0]):int(boundRect[i][0]+boundRect[i][2])]
            cards.append(card)
    return cards

def find_5_cards(image, common_markers, card_dim):
    crop = image[-image.shape[0]//3:-20,10:-10]#get bottom third of image
    final_cards = []
    best_idx = (0,0)
    best_len = 0
    for sig in range (2,5):
        cards = find_cards(crop,sig, card_dim)
        #If we find 5 cards, return results
        if len(cards)==5 :
            return cards
    
    #If we don't have 5 cards, try loris' method
    if(best_len!=5):
        new_img = isolate_common_cards(crop)
        for sig in range (2,5):
            cards = find_cards(new_img, sig, card_dim)
            #If we find 5 cards, record sig and exit
            if len(cards)==5 :
                return cards
    #If both methods don't return good results, return basic separation (/5)
    return find_common_search_area_v1(image, common_markers, card_dim)

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

def get_predictors(cards, lower, upper):
    #bin_cards=[]
    #for card in cards:
    #    bin_cards.append(binarization(card))
    sobel_test_images = edge_detector(cards)
    #test_contours = one_contour_by_img(sobel_test_images)
    test_all_contours = contours_by_img(sobel_test_images)
    test_filtered_contours = filter_contours_by_size(test_all_contours, lower, upper)
    cards_comp_ct = [complex_contours(card_cont) for card_cont in test_filtered_contours]
    test_descr = [n_FT_descr(comp_ct, 10) for comp_ct in cards_comp_ct]
    test_3D_descr = []
    for card_descr in test_descr:
        if not card_descr.size==0:
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
